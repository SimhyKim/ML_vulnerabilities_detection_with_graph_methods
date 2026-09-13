from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import zlib
import torch
from torch_geometric.data import Data, Dataset

from cpg_vuln.cpg_cache import load_or_build_cpg
from cpg_vuln.cpg_extractor import CPGExtractor
from cpg_vuln.domain_knowledge import (
    API_WHITELIST,
    DOMAIN_GRAPH_DIM,
    classify_api,
    graph_domain_vector,
)
from cpg_vuln.motifs import MOTIF_DIM, extract_motifs
from cpg_vuln.slicer import extract_spg


NUM_TYPE_BUCKETS = 512
NUM_TOKEN_BUCKETS = 2048

EDGE_KIND_TO_ID = {
    "AST": 0,
    "AST_REV": 1,
    "SEQ": 2,
    "SEQ_REV": 3,
    "DFG": 4,
    "DFG_REV": 5,
    "CALL_CALLEE": 6,
    "CALL_CALLEE_REV": 7,
    "CALL_ARG": 8,
    "CALL_ARG_REV": 9,
    "CTRL_DEP": 10,
    "CTRL_DEP_REV": 11,
    "RET_FLOW": 12,
    "RET_FLOW_REV": 13,
    "UNKNOWN": 14,
}

NODE_FLAG_DIM = 15


def _normalize_token(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    if not text:
        return ""
    if text.isidentifier():
        if text in API_WHITELIST:
            return text
        return "ID"
    num_text = text.replace(".", "", 1).replace("_", "")
    if num_text.isdigit():
        return "NUM"
    if text[0] in {'"', "'"}:
        return "STR"
    if len(text) <= 3:
        return text
    return text


def _hash_bucket(value: str, buckets: int) -> int:
    if not value:
        return 0
    return zlib.crc32(value.encode("utf-8")) % buckets


def _node_flags(node_type: str, norm_token: str, raw_text: str, depth: int, child_pos: int) -> torch.Tensor:
    t = node_type
    is_id = 1.0 if t == "identifier" else 0.0
    is_lit = 1.0 if ("literal" in t or norm_token in {"NUM", "STR"}) else 0.0
    is_call = 1.0 if "call" in t else 0.0
    is_ctrl = 1.0 if t in {"if_statement", "for_statement", "while_statement", "switch_statement"} else 0.0
    is_assign = 1.0 if t in {"assignment_expression", "init_declarator"} else 0.0
    is_ret = 1.0 if t == "return_statement" else 0.0
    is_api = 1.0 if norm_token in API_WHITELIST else 0.0
    has_ptr_op = 1.0 if ("->" in raw_text or "*" in raw_text or "&" in raw_text) else 0.0
    depth_norm = min(float(depth), 40.0) / 40.0
    child_pos_norm = min(float(child_pos), 20.0) / 20.0
    api = classify_api(norm_token)
    return torch.tensor(
        [
            is_id,
            is_lit,
            is_call,
            is_ctrl,
            is_assign,
            is_ret,
            is_api,
            has_ptr_op,
            depth_norm,
            child_pos_norm,
            api["is_memory_sink"],
            api["is_alloc"],
            api["is_injection"],
            api["is_source"],
            api["is_any_sec_api"],
        ],
        dtype=torch.float32,
    )


def _item_code_label(item: Dict[str, Any]) -> tuple[str, float]:
    if "func_before" in item:
        code = item["func_before"]
        y = float(item.get("vul", item.get("label", 0)))
    else:
        code = item.get("code") or item.get("func") or item.get("function") or ""
        y = float(item.get("label", item.get("vul", 0)))
    return str(code or ""), y


class CPGVulnDataset(Dataset):

    def __init__(
        self,
        hf_split,
        enabled_edge_kinds: Optional[Set[str]] = None,
        use_slice: bool = False,
        slice_hops: int = 2,
        use_domain_graph_feat: bool = True,
        use_motifs: bool = True,
        cache_graphs: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        super().__init__()
        self.data = hf_split
        self.cpg_extractor = CPGExtractor()
        self.enabled_edge_kinds = enabled_edge_kinds
        self.use_slice = use_slice
        self.slice_hops = slice_hops
        self.use_domain_graph_feat = use_domain_graph_feat
        self.use_motifs = use_motifs
        self.cache_graphs = cache_graphs
        self.cache_dir = cache_dir
        self._cache: OrderedDict[int, Data] = OrderedDict()
        self._ram_limit = 256

    def _build_graph(self, item: Dict[str, Any]) -> Data:
        code, y_val = _item_code_label(item)
        func_hash = str(item.get("hash", "") or "")
        if self.cache_dir is None:
            cpg = load_or_build_cpg(code, func_hash, self.cpg_extractor)
        else:
            cpg = load_or_build_cpg(code, func_hash, self.cpg_extractor, cache_dir=self.cache_dir)
        if self.use_slice:
            cpg = extract_spg(cpg, hops=self.slice_hops)

        type_ids: List[int] = []
        token_ids: List[int] = []
        flags: List[torch.Tensor] = []
        id2idx: Dict[int, int] = {}
        node_types: List[str] = []
        norm_tokens: List[str] = []
        raw_texts: List[str] = []

        for idx, node in enumerate(cpg["nodes"]):
            node_id = int(node["id"])
            id2idx[node_id] = idx

            node_type = str(node.get("type", "UNKNOWN"))
            raw_text = str(node.get("code") or node.get("name") or "")
            norm_token = _normalize_token(raw_text)
            depth = int(node.get("depth", 0))
            child_pos = int(node.get("child_pos", 0))

            type_ids.append(_hash_bucket(node_type, NUM_TYPE_BUCKETS))
            token_ids.append(_hash_bucket(norm_token, NUM_TOKEN_BUCKETS))
            flags.append(_node_flags(node_type, norm_token, raw_text, depth, child_pos))
            node_types.append(node_type)
            norm_tokens.append(norm_token)
            raw_texts.append(raw_text)

        type_ids_t = torch.tensor(type_ids, dtype=torch.long)
        token_ids_t = torch.tensor(token_ids, dtype=torch.long)
        flags_t = (
            torch.stack(flags, dim=0)
            if flags
            else torch.zeros((0, NODE_FLAG_DIM), dtype=torch.float32)
        )

        edges_src, edges_dst = [], []
        edge_types: List[int] = []
        for e in cpg["edges"]:
            kind = str(e.get("kind", "UNKNOWN"))
            if self.enabled_edge_kinds is not None and kind not in self.enabled_edge_kinds:
                continue
            src = id2idx[int(e["src"])]
            dst = id2idx[int(e["dst"])]
            edges_src.append(src)
            edges_dst.append(dst)
            edge_types.append(EDGE_KIND_TO_ID.get(kind, EDGE_KIND_TO_ID["UNKNOWN"]))

        if len(type_ids) == 0:
            type_ids_t = torch.tensor([0], dtype=torch.long)
            token_ids_t = torch.tensor([0], dtype=torch.long)
            flags_t = torch.zeros((1, NODE_FLAG_DIM), dtype=torch.float32)
            edges_src = [0]
            edges_dst = [0]
            edge_types = [EDGE_KIND_TO_ID["UNKNOWN"]]
            node_types = ["UNKNOWN"]
            norm_tokens = [""]
            raw_texts = [""]

        if len(edges_src) == 0:
            edges_src = [0]
            edges_dst = [0]
            edge_types = [EDGE_KIND_TO_ID["UNKNOWN"]]

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        y_bin = torch.tensor(y_val, dtype=torch.float32)

        if self.use_domain_graph_feat:
            domain = torch.tensor(
                graph_domain_vector(node_types, norm_tokens, raw_texts),
                dtype=torch.float32,
            ).view(1, -1)
        else:
            domain = torch.zeros((1, DOMAIN_GRAPH_DIM), dtype=torch.float32)

        if self.use_motifs:
            motif = torch.tensor(extract_motifs(cpg), dtype=torch.float32).view(1, -1)
            if motif.numel() != MOTIF_DIM:
                motif = torch.zeros((1, MOTIF_DIM), dtype=torch.float32)
        else:
            motif = torch.zeros((1, MOTIF_DIM), dtype=torch.float32)

        data = Data(
            type_ids=type_ids_t,
            token_ids=token_ids_t,
            flags=flags_t,
            edge_index=edge_index,
            edge_type=edge_type,
            domain=domain,
            motif=motif,
            y=y_bin,
        )
        data.num_nodes = int(type_ids_t.numel())
        return data

    def len(self) -> int:
        return len(self.data)

    def get(self, idx: int) -> Data:
        if self.cache_graphs and idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        item = self.data[idx]
        data = self._build_graph(item)
        if self.cache_graphs:
            self._cache[idx] = data
            self._cache.move_to_end(idx)
            while len(self._cache) > self._ram_limit:
                self._cache.popitem(last=False)
        return data
