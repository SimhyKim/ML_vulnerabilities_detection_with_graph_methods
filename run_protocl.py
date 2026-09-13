
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as GraphDataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpg_vuln import graph_dataset as gd
from cpg_vuln.cpg_cache import DEFAULT_CACHE_DIR, load_or_build_cpg
from cpg_vuln.cpg_extractor import CPGExtractor
from cpg_vuln.domain_knowledge import graph_domain_vector
from cpg_vuln.graph_augment import dual_vulnerability_views
from cpg_vuln.graph_dataset import (
    CPGVulnDataset,
    EDGE_KIND_TO_ID,
    NODE_FLAG_DIM,
    NUM_TOKEN_BUCKETS,
    NUM_TYPE_BUCKETS,
)
from cpg_vuln.model_protocl import VulProtoCL
from cpg_vuln.motifs import MOTIF_DIM, extract_motifs
from cpg_vuln.progress import fallen, progress
from cpg_vuln.slicer import extract_spg
from vulprotocl.seq import BiLSTM, SeqDS, predict_seq, train_bilstm
from vulprotocl.data import (
    default_reveal_dir,
    load_reveal,
    stratified_split,
    subsample,
    write_split_hashes,
)
from vulprotocl.metrics import (
    FocalLoss,
    best_threshold,
    metrics_from_probs,
    set_seed,
    sigmoid,
)


def cpg_to_data(cpg: dict, y: float, use_motifs: bool = True) -> Data:
    type_ids, token_ids, flags = [], [], []
    node_types, norm_tokens, raw_texts = [], [], []
    id2idx: Dict[int, int] = {}
    for idx, node in enumerate(cpg.get("nodes", [])):
        nid = int(node["id"])
        id2idx[nid] = idx
        node_type = str(node.get("type", "UNKNOWN"))
        raw_text = str(node.get("code") or node.get("name") or "")
        norm_token = gd._normalize_token(raw_text)
        depth = int(node.get("depth", 0))
        child_pos = int(node.get("child_pos", 0))
        type_ids.append(gd._hash_bucket(node_type, gd.NUM_TYPE_BUCKETS))
        token_ids.append(gd._hash_bucket(norm_token, gd.NUM_TOKEN_BUCKETS))
        flags.append(gd._node_flags(node_type, norm_token, raw_text, depth, child_pos))
        node_types.append(node_type)
        norm_tokens.append(norm_token)
        raw_texts.append(raw_text)

    if not type_ids:
        type_ids = [0]
        token_ids = [0]
        flags = [torch.zeros(gd.NODE_FLAG_DIM)]
        node_types, norm_tokens, raw_texts = ["UNKNOWN"], [""], [""]
        id2idx = {0: 0}

    edges_src, edges_dst, edge_types = [], [], []
    for e in cpg.get("edges", []):
        s, d = int(e["src"]), int(e["dst"])
        if s not in id2idx or d not in id2idx:
            continue
        edges_src.append(id2idx[s])
        edges_dst.append(id2idx[d])
        edge_types.append(
            gd.EDGE_KIND_TO_ID.get(str(e.get("kind", "UNKNOWN")), gd.EDGE_KIND_TO_ID["UNKNOWN"])
        )
    if not edges_src:
        edges_src, edges_dst, edge_types = [0], [0], [gd.EDGE_KIND_TO_ID["UNKNOWN"]]

    domain = torch.tensor(
        graph_domain_vector(node_types, norm_tokens, raw_texts), dtype=torch.float32
    ).view(1, -1)
    if use_motifs:
        motif = torch.tensor(extract_motifs(cpg), dtype=torch.float32).view(1, -1)
        if motif.numel() != MOTIF_DIM:
            motif = torch.zeros((1, MOTIF_DIM), dtype=torch.float32)
    else:
        motif = torch.zeros((1, MOTIF_DIM), dtype=torch.float32)
    data = Data(
        type_ids=torch.tensor(type_ids, dtype=torch.long),
        token_ids=torch.tensor(token_ids, dtype=torch.long),
        flags=torch.stack(flags),
        edge_index=torch.tensor([edges_src, edges_dst], dtype=torch.long),
        edge_type=torch.tensor(edge_types, dtype=torch.long),
        domain=domain,
        motif=motif,
        y=torch.tensor(float(y), dtype=torch.float32),
    )
    data.num_nodes = int(data.type_ids.numel())
    return data


class DualViewDataset(Dataset):
    def __init__(
        self,
        hf_split,
        seed: int = 42,
        dual_view: bool = True,
        use_motifs: bool = True,
        cache_dir: Path | None = None,
    ):
        self.data = hf_split
        self.extractor = CPGExtractor()
        self.base_seed = seed
        self.dual_view = dual_view
        self.use_motifs = use_motifs
        self.cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
        self._n_get = 0

    def __len__(self):
        return len(self.data)

    def _full(self, idx: int) -> dict:
        row = self.data[idx]
        code = row["func_before"]
        func_hash = str(row.get("hash", "") or "")
        return load_or_build_cpg(code, func_hash, self.extractor, cache_dir=self.cache_dir)

    def __getitem__(self, idx: int):
        self._n_get += 1
        if self._n_get == 1 or self._n_get % 100 == 0:
            progress(
                f"DualViewDataset __getitem__ #{self._n_get}/{len(self)} idx={idx} "
                f"(first epoch is slow: tree-sitter + SPG views)"
            )
        try:
            y = float(self.data[idx]["vul"])
            full = self._full(idx)
            if not self.dual_view:
                g = extract_spg(full, hops=2)
                d = cpg_to_data(g, y, use_motifs=self.use_motifs)
                return d, d
            rng = random.Random(self.base_seed * 1000003 + idx)
            va, vb = dual_vulnerability_views(full, rng=rng)
            return (
                cpg_to_data(va, y, use_motifs=self.use_motifs),
                cpg_to_data(vb, y, use_motifs=self.use_motifs),
            )
        except Exception as err:
            fallen(f"DualViewDataset.__getitem__ idx={idx}", err)
            raise


def collate_dual(batch):
    a = Batch.from_data_list([x[0] for x in batch])
    b = Batch.from_data_list([x[1] for x in batch])
    return a, b


@torch.no_grad()
def predict_protocl(model, loader, device):
    model.eval()
    logits, ys = [], []
    n = 0
    for batch in loader:
        n += 1
        if n == 1 or n % 50 == 0:
            progress(f"predict_protocl batch {n}")
        try:
            batch = batch.to(device)
            _, logit, _ = model(batch)
            logits.extend(logit.cpu().numpy().tolist())
            ys.extend(batch.y.cpu().numpy().tolist())
        except Exception as err:
            fallen(f"predict_protocl batch={n}", err)
            raise
    return np.array(logits), np.array(ys)


def train_protocl(
    train,
    val,
    device,
    epochs,
    batch_size,
    seed,
    lambda_proto=0.3,
    lambda_view=0.2,
    dual_view=True,
    use_motifs=True,
    cache_graphs=True,
    cache_dir: Path | None = None,
):
    progress(
        f"train_protocl start n_train={len(train)} n_val={len(val)} "
        f"batch={batch_size} epochs={epochs} dual_view={dual_view}",
        stage="protocl_train_setup",
    )
    set_seed(seed)
    tr_ds = DualViewDataset(
        train,
        seed=seed,
        dual_view=dual_view,
        use_motifs=use_motifs,
        cache_dir=cache_dir,
    )
    va_ds = CPGVulnDataset(
        val,
        use_slice=True,
        slice_hops=2,
        use_domain_graph_feat=True,
        use_motifs=use_motifs,
        cache_graphs=cache_graphs,
        cache_dir=cache_dir,
    )

    y_train = np.array(train["vul"])
    num_pos = max(int(y_train.sum()), 1)
    num_neg = max(len(y_train) - num_pos, 1)
    weights = np.where(y_train == 1, float(num_neg) / num_pos, 1.0)
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double), len(weights), True
    )

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate_dual)
    va_loader = GraphDataLoader(va_ds, batch_size=batch_size, shuffle=False)
    n_batches = max(1, (len(tr_ds) + batch_size - 1) // batch_size)
    progress(f"loaders ready ~{n_batches} train batches/epoch (first epoch parses CPGs)", stage="protocl_train")

    model = build_protocl_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-2)
    crit = FocalLoss(torch.tensor([num_neg / num_pos], device=device), gamma=2.0)
    progress(f"model on {device} params={sum(p.numel() for p in model.parameters())}")

    best_state, best_f1, best_epoch = None, -1.0, 0
    for ep in range(epochs):
        model.train()
        total = 0.0
        n = 0
        progress(f"epoch {ep+1}/{epochs} waiting for first batch…", stage=f"protocl_epoch_{ep+1}")
        for view_a, view_b in tr_loader:
            if n == 0:
                progress(f"epoch {ep+1} got first batch — GPU step starting")
            try:
                view_a = view_a.to(device)
                y = view_a.y
                _, logit_a, z_a = model(view_a)
                if dual_view:
                    view_b = view_b.to(device)
                    _, logit_b, z_b = model(view_b)
                    loss_cls = 0.5 * (crit(logit_a, y) + crit(logit_b, y))
                    loss_proto = (
                        0.5
                        * (model.prototype_loss(z_a, y) + model.prototype_loss(z_b, y))
                        if lambda_proto > 0
                        else logit_a.new_zeros(())
                    )
                    loss_view = (
                        model.view_consistency_loss(z_a, z_b)
                        if lambda_view > 0
                        else logit_a.new_zeros(())
                    )
                else:
                    loss_cls = crit(logit_a, y)
                    loss_proto = (
                        model.prototype_loss(z_a, y)
                        if lambda_proto > 0
                        else logit_a.new_zeros(())
                    )
                    loss_view = logit_a.new_zeros(())
                loss = loss_cls + lambda_proto * loss_proto + lambda_view * loss_view
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            except Exception as err:
                fallen(f"protocl train epoch={ep+1} batch={n+1}/{n_batches}", err)
                raise
            total += float(loss.item())
            n += 1
            if n <= 5 or n % 20 == 0 or n == n_batches:
                progress(
                    f"epoch {ep+1}/{epochs} batch {n}/{n_batches} "
                    f"loss={loss.item():.4f} avg={total/n:.4f} "
                    f"cls={float(loss_cls):.3f} proto={float(loss_proto):.3f} view={float(loss_view):.3f}"
                )

        progress(f"epoch {ep+1} train done; running val GNN…", stage=f"protocl_val_{ep+1}")
        logits, ys = predict_protocl(model, va_loader, device)
        _, f1 = best_threshold(ys, sigmoid(logits))
        print(
            f"[protocl] epoch {ep+1}/{epochs} loss={total/max(n,1):.4f} val_f1={f1:.4f}",
            flush=True,
        )
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            progress(f"new best val_f1={best_f1:.4f} at epoch {best_epoch}")

    print(f"[protocl] best_epoch={best_epoch} best_val_f1={best_f1:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("ProtoCL never stored a checkpoint (no successful train batch?)")
    model.load_state_dict(best_state)
    return model, best_epoch


def build_protocl_model(device: torch.device) -> VulProtoCL:
    return VulProtoCL(
        NUM_TYPE_BUCKETS,
        NUM_TOKEN_BUCKETS,
        max(EDGE_KIND_TO_ID.values()) + 1,
        node_flag_dim=NODE_FLAG_DIM,
        hidden_dim=160,
        num_layers=3,
        num_prototypes=4,
    ).to(device)


def ablation_tag(args) -> str:
    parts = []
    if getattr(args, "no_dual_view", False):
        parts.append("nodual")
    if getattr(args, "no_prototypes", False):
        parts.append("noproto")
    if getattr(args, "no_motifs", False):
        parts.append("nomotif")
    if getattr(args, "no_fusion", False):
        parts.append("nofusion")
    return "_".join(parts) if parts else "full"


def run_fingerprint(args, n_train: int, n_val: int, n_test: int, tag: str) -> dict:
    return {
        "seed": int(args.seed),
        "ablation": tag,
        "max_train": int(args.max_train),
        "max_val": int(args.max_val),
        "max_test": int(args.max_test),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "lambda_proto": float(args.lambda_proto),
        "lambda_view": float(args.lambda_view),
        "no_dual_view": bool(args.no_dual_view),
        "no_prototypes": bool(args.no_prototypes),
        "no_motifs": bool(args.no_motifs),
        "no_fusion": bool(args.no_fusion),
        "max_len": int(args.max_len),
        "dataset": str(getattr(args, "dataset", "reveal")),
    }


def resolve_ckpt_dir(args, publish_dir: Path | None, tag: str) -> Path:
    explicit = str(getattr(args, "ckpt_dir", "") or "").strip()
    if explicit:
        return Path(explicit)
    if publish_dir is not None:
        return Path(publish_dir)
    return ROOT / "results" / "ckpt" / f"seed{args.seed}_{tag}"


def _coerce_same(stored, current) -> bool:
    if stored is None:
        return True
    if isinstance(current, bool):
        return bool(stored) == current
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        try:
            return abs(float(stored) - float(current)) < 1e-9
        except (TypeError, ValueError):
            return False
    return stored == current


def blob_fingerprint(blob: dict) -> dict:
    fp = dict(blob.get("fingerprint") or {})
    stored_args = blob.get("args")
    if isinstance(stored_args, dict):
        for key in (
            "seed",
            "max_train",
            "max_val",
            "max_test",
            "n_train",
            "n_val",
            "n_test",
            "lambda_proto",
            "lambda_view",
            "no_dual_view",
            "no_prototypes",
            "no_motifs",
            "no_fusion",
            "max_len",
            "dataset",
        ):
            if key not in fp and key in stored_args:
                fp[key] = stored_args[key]
    for key in ("n_train", "n_val", "n_test", "ablation", "best_epoch"):
        if key not in fp and key in blob:
            fp[key] = blob[key]
    return fp


def fingerprint_compatible(blob: dict, current: dict) -> bool:
    stored = blob_fingerprint(blob)
    required = (
        "seed",
        "max_train",
        "max_val",
        "max_test",
        "lambda_proto",
        "lambda_view",
        "no_dual_view",
        "no_prototypes",
        "no_motifs",
        "no_fusion",
        "max_len",
    )
    for key in required:
        if key in stored and not _coerce_same(stored[key], current[key]):
            return False
    if "ablation" in stored and stored["ablation"] != current["ablation"]:
        return False
    if "dataset" in stored and stored["dataset"] != current.get("dataset", "reveal"):
        return False
    for key in ("n_train", "n_val", "n_test"):
        if key in stored and stored[key] is not None:
            if not _coerce_same(stored[key], current[key]):
                return False
    return True


def atomic_torch_save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save_stage_ckpt(path: Path, payload: dict, fingerprint: dict) -> None:
    payload = dict(payload)
    payload["fingerprint"] = fingerprint
    atomic_torch_save(path, payload)
    progress(f"saved {path}")


def load_compatible_ckpt(paths: list[Path], current: dict) -> dict | None:
    for path in paths:
        if not path.is_file():
            continue
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as err:
            progress(f"ignore unreadable ckpt {path}: {err}")
            continue
        if not isinstance(blob, dict):
            progress(f"ignore non-dict ckpt {path}")
            continue
        if not fingerprint_compatible(blob, current):
            progress(f"ignore incompatible ckpt {path}")
            continue
        progress(f"resume from {path}")
        return blob
    return None


def rows_complete(rows: list[dict], current: dict, no_fusion: bool) -> list[dict] | None:
    if not rows:
        return None
    try:
        n_train = int(float(rows[0]["n_train"]))
    except (KeyError, TypeError, ValueError):
        return None
    if n_train != int(current["n_train"]):
        return None
    configs = {r.get("config") for r in rows}
    need = {"protocl_gnn"}
    if not no_fusion:
        need |= {"bilstm_ref", "protocl_late_fusion"}
    if not need <= configs:
        return None
    return rows


def metrics_complete(path: Path, current: dict, no_fusion: bool) -> list[dict] | None:
    if path.suffix.lower() == ".json":
        if not path.is_file():
            return None
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(rows, list):
            return None
        return rows_complete(rows, current, no_fusion)
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows_complete(rows, current, no_fusion)


def csv_fieldnames(rows_to_write: list[dict]) -> list[str]:
    preferred = [
        "config",
        "dataset",
        "ablation",
        "seed",
        "best_epoch",
        "n_train",
        "n_val",
        "n_test",
        "alpha",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "balanced_acc",
        "prec_at_rec_0.2",
        "prec_at_rec_0.4",
        "prec_at_rec_0.6",
    ]
    present: list[str] = []
    seen: set[str] = set()
    for rec in rows_to_write:
        for key in rec.keys():
            if key not in seen:
                seen.add(key)
                present.append(key)
    return [k for k in preferred if k in seen] + [k for k in present if k not in preferred]


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = csv_fieldnames(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


def write_metrics_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)


def try_load_gnn(ckpt_dir: Path, current: dict, device: torch.device):
    blob = load_compatible_ckpt(
        [ckpt_dir / "ckpt_gnn.pt", ckpt_dir / "protocl_gnn_partial.pt"],
        current,
    )
    if blob is None or "protocl" not in blob:
        return None
    model = build_protocl_model(device)
    try:
        model.load_state_dict(blob["protocl"])
    except Exception as err:
        progress(f"GNN state_dict mismatch, retraining: {err}")
        return None
    best_epoch = int(blob.get("best_epoch") or 0)
    return model, best_epoch


def try_load_bilstm(ckpt_dir: Path, current: dict, device: torch.device):
    blob = load_compatible_ckpt(
        [ckpt_dir / "ckpt_bilstm.pt", ckpt_dir / "bilstm_partial.pt"],
        current,
    )
    if blob is None or "bilstm" not in blob or "vocab" not in blob:
        return None
    vocab = blob["vocab"]
    model = BiLSTM(len(vocab)).to(device)
    try:
        model.load_state_dict(blob["bilstm"])
    except Exception as err:
        progress(f"BiLSTM state_dict mismatch, retraining: {err}")
        return None
    return model, vocab


def try_load_scores(ckpt_dir: Path, current: dict, need_seq: bool = False) -> dict | None:
    blob = load_compatible_ckpt([ckpt_dir / "ckpt_scores.pt"], current)
    if blob is None:
        return None
    need = ("va_pg", "te_pg", "va_y", "te_y")
    if any(k not in blob for k in need):
        return None
    if len(blob["va_y"]) != int(current["n_val"]) or len(blob["te_y"]) != int(current["n_test"]):
        progress("ignore ckpt_scores.pt: split size mismatch")
        return None
    if need_seq and any(k not in blob for k in ("va_ps", "te_ps")):
        return blob
    return blob


def run_protocl_once(
    args,
    publish_dir: Path | None = None,
    splits: tuple | None = None,
) -> list:
    dual_view = not args.no_dual_view
    use_motifs = not args.no_motifs
    lambda_proto = 0.0 if args.no_prototypes else args.lambda_proto
    lambda_view = 0.0 if args.no_dual_view else args.lambda_view
    tag = ablation_tag(args)
    force = bool(getattr(args, "force_retrain", False))
    dataset = str(getattr(args, "dataset", "reveal"))

    progress(f"seed={args.seed} start ablation={tag} dataset={dataset}", stage="seed_start")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ablation={tag} seed={args.seed} dataset={dataset}", flush=True)

    if splits is not None:
        train, val, test = splits
        train = subsample(train, args.max_train, args.seed)
        val = subsample(val, args.max_val, args.seed + 1)
        test = subsample(test, args.max_test, args.seed + 2)
    else:
        progress("loading ReVeal JSON…", stage="load_reveal")
        full = load_reveal(default_reveal_dir())
        progress(f"loaded n={len(full)} pos={int(sum(full['vul']))}")
        train, val, test = stratified_split(full, args.seed)
        train = subsample(train, args.max_train, args.seed)
        val = subsample(val, args.max_val, args.seed + 1)
        test = subsample(test, args.max_test, args.seed + 2)
    print(
        f"splits train={len(train)} val={len(val)} test={len(test)} "
        f"pos_train={int(sum(train['vul']))}",
        flush=True,
    )

    if dataset == "primevul":
        split_dir = ROOT / "results" / "primevul" / "splits"
        split_prefix = "primevul"
    else:
        split_dir = ROOT / "results" / "splits"
        split_prefix = "reveal"
    cap = "" if int(args.max_train) <= 0 else f"_cap{args.max_train}"
    write_split_hashes(train, split_dir / f"{split_prefix}_seed{args.seed}_train{cap}.txt")
    write_split_hashes(val, split_dir / f"{split_prefix}_seed{args.seed}_val{cap}.txt")
    write_split_hashes(test, split_dir / f"{split_prefix}_seed{args.seed}_test{cap}.txt")

    fp = run_fingerprint(args, len(train), len(val), len(test), tag)
    ckpt_dir = resolve_ckpt_dir(args, publish_dir, tag)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    progress(f"ckpt_dir={ckpt_dir} force_retrain={force}")

    if not force:
        done = metrics_complete(ckpt_dir / "metrics.csv", fp, args.no_fusion)
        if done is None:
            done = metrics_complete(ckpt_dir / "metrics.json", fp, args.no_fusion)
        if done is not None:
            progress("metrics already match this split — skip training", stage="resume")
            for r in done:
                print(r, flush=True)
            return done

    cache_graphs = not bool(getattr(args, "no_cache_graphs", False))
    explicit_cache = str(getattr(args, "cpg_cache_dir", "") or "").strip()
    if explicit_cache:
        cpg_cache = Path(explicit_cache)
    elif dataset == "primevul":
        cpg_cache = ROOT / "datasets" / "primevul_cpg_cache"
    else:
        cpg_cache = DEFAULT_CACHE_DIR

    def gnn_loader(split):
        ds = CPGVulnDataset(
            split,
            use_slice=True,
            slice_hops=2,
            use_domain_graph_feat=True,
            use_motifs=use_motifs,
            cache_graphs=cache_graphs,
            cache_dir=cpg_cache,
        )
        return GraphDataLoader(ds, batch_size=args.batch_gnn, shuffle=False)

    gnn = None
    best_epoch = 0
    loaded_gnn = None if force else try_load_gnn(ckpt_dir, fp, device)
    if loaded_gnn is not None:
        gnn, best_epoch = loaded_gnn
        progress(f"loaded GNN weights best_epoch={best_epoch}", stage="resume")
    else:
        progress("starting ProtoCL GNN training (longest stage)", stage="protocl_train")
        gnn, best_epoch = train_protocl(
            train,
            val,
            device,
            args.epochs_gnn,
            args.batch_gnn,
            args.seed,
            lambda_proto=lambda_proto,
            lambda_view=lambda_view,
            dual_view=dual_view,
            use_motifs=use_motifs,
            cache_graphs=cache_graphs,
            cache_dir=cpg_cache,
        )
        save_stage_ckpt(
            ckpt_dir / "ckpt_gnn.pt",
            {
                "protocl": gnn.state_dict(),
                "best_epoch": best_epoch,
                "args": vars(args),
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
            },
            fp,
        )

    scores = None if force else try_load_scores(ckpt_dir, fp, need_seq=not args.no_fusion)
    if scores is not None and all(k in scores for k in ("va_pg", "te_pg", "va_y", "te_y")):
        va_pg = np.asarray(scores["va_pg"])
        te_pg = np.asarray(scores["te_pg"])
        va_y = np.asarray(scores["va_y"])
        te_y = np.asarray(scores["te_y"])
        progress("loaded GNN val/test scores", stage="resume")
    else:
        progress("ProtoCL test/val GNN forward", stage="protocl_eval")
        va_g = predict_protocl(gnn, gnn_loader(val), device)
        progress("val GNN logits done; scoring test")
        te_g = predict_protocl(gnn, gnn_loader(test), device)
        va_pg, va_y = sigmoid(va_g[0]), va_g[1]
        te_pg, te_y = sigmoid(te_g[0]), te_g[1]
        scores = {
            "va_pg": va_pg,
            "te_pg": te_pg,
            "va_y": va_y,
            "te_y": te_y,
            "best_epoch": best_epoch,
            "args": vars(args),
        }
        save_stage_ckpt(ckpt_dir / "ckpt_scores.pt", scores, fp)

    th_g = best_threshold(va_y, va_pg)[0]
    rows = [
        {
            "config": "protocl_gnn",
            "ablation": tag,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            **metrics_from_probs(te_y, te_pg, th_g),
        }
    ]

    fusion_ckpt: dict
    if not args.no_fusion:
        loaded_seq = None if force else try_load_bilstm(ckpt_dir, fp, device)
        if loaded_seq is not None:
            bilstm, vocab = loaded_seq
            progress("loaded BiLSTM weights", stage="resume")
        else:
            progress("training BiLSTM (late fusion)", stage="bilstm_train")
            bilstm, vocab = train_bilstm(
                train, val, device, args.epochs_seq, args.batch_seq, args.max_len, args.seed
            )
            save_stage_ckpt(
                ckpt_dir / "ckpt_bilstm.pt",
                {
                    "bilstm": bilstm.state_dict(),
                    "vocab": vocab,
                    "args": vars(args),
                    "n_train": len(train),
                    "n_val": len(val),
                    "n_test": len(test),
                },
                fp,
            )

        def seq_loader(split):
            ds = SeqDS(list(split["func_before"]), np.array(split["vul"]), vocab, args.max_len)
            return DataLoader(ds, batch_size=args.batch_seq, shuffle=False)

        if scores is not None and "va_ps" in scores and "te_ps" in scores:
            va_ps = np.asarray(scores["va_ps"])
            te_ps = np.asarray(scores["te_ps"])
            progress("loaded BiLSTM val/test scores", stage="resume")
        else:
            va_s = predict_seq(bilstm, seq_loader(val), device)
            te_s = predict_seq(bilstm, seq_loader(test), device)
            va_ps, te_ps = sigmoid(va_s[0]), sigmoid(te_s[0])
            scores = dict(scores or {})
            scores.update(
                {
                    "va_ps": va_ps,
                    "te_ps": te_ps,
                    "best_epoch": best_epoch,
                    "args": vars(args),
                }
            )
            save_stage_ckpt(ckpt_dir / "ckpt_scores.pt", scores, fp)

        th_s = best_threshold(va_y, va_ps)[0]
        progress("BiLSTM done; blending alpha on val", stage="fusion_tune")
        rows.append(
            {
                "config": "bilstm_ref",
                "ablation": tag,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                **metrics_from_probs(te_y, te_ps, th_s),
            }
        )

        best = {"f1": -1.0, "alpha": 1.0, "th": 0.5}
        for alpha in np.linspace(0.0, 1.0, 21):
            blended = alpha * va_ps + (1 - alpha) * va_pg
            th, f1 = best_threshold(va_y, blended)
            if f1 > best["f1"]:
                best = {"f1": f1, "alpha": float(alpha), "th": float(th)}
        print(
            f"Best ProtoCL fusion alpha={best['alpha']:.2f} val_f1={best['f1']:.4f} th={best['th']:.2f}",
            flush=True,
        )
        rows.append(
            {
                "config": "protocl_late_fusion",
                "ablation": tag,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "alpha": best["alpha"],
                "threshold": best["th"],
                **metrics_from_probs(
                    te_y,
                    best["alpha"] * te_ps + (1 - best["alpha"]) * te_pg,
                    best["th"],
                ),
            }
        )
        fusion_ckpt = {
            "protocl": gnn.state_dict(),
            "bilstm": bilstm.state_dict(),
            "vocab": vocab,
            "alpha": best["alpha"],
            "threshold": best["th"],
            "best_epoch": best_epoch,
            "args": vars(args),
            "fingerprint": fp,
        }
    else:
        fusion_ckpt = {
            "protocl": gnn.state_dict(),
            "best_epoch": best_epoch,
            "args": vars(args),
            "fingerprint": fp,
        }

    for r in rows:
        r["dataset"] = dataset
        print(r, flush=True)

    progress(f"writing metrics/checkpoints seed={args.seed}", stage="save")
    write_metrics_json(ckpt_dir / "metrics.json", rows)
    atomic_torch_save(ckpt_dir / "protocl_best.pt", fusion_ckpt)
    write_metrics_csv(ckpt_dir / "metrics.csv", rows)

    if dataset == "reveal":
        out = ROOT / "results" / "runs"
        stem = f"results_reveal_protocl_{tag}_seed{args.seed}"
        default_csv = "results_reveal_protocl.csv"
    else:
        out = ROOT / "results" / "runs" / dataset
        stem = f"results_{dataset}_protocl_{tag}_seed{args.seed}"
        default_csv = f"results_{dataset}_protocl.csv"
    out.mkdir(exist_ok=True)
    write_metrics_json(out / f"{stem}.json", rows)
    atomic_torch_save(out / f"protocl_{dataset}_{tag}_seed{args.seed}_best.pt", fusion_ckpt)
    write_metrics_csv(out / f"{stem}.csv", rows)
    if tag == "full":
        write_metrics_csv(out / default_csv, rows)

    if publish_dir is not None:
        publish_dir.mkdir(parents=True, exist_ok=True)
        if publish_dir.resolve() != ckpt_dir.resolve():
            write_metrics_json(publish_dir / "metrics.json", rows)
            atomic_torch_save(publish_dir / "protocl_best.pt", fusion_ckpt)
            write_metrics_csv(publish_dir / "metrics.csv", rows)
        meta = {
            "seed": args.seed,
            "dataset": dataset,
            "ablation": tag,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "best_epoch": best_epoch,
            "args": vars(args),
            "ckpt_dir": str(ckpt_dir),
            "cpg_cache": str(cpg_cache),
            "split_files": {
                "train": str(split_dir / f"{split_prefix}_seed{args.seed}_train{cap}.txt"),
                "val": str(split_dir / f"{split_prefix}_seed{args.seed}_val{cap}.txt"),
                "test": str(split_dir / f"{split_prefix}_seed{args.seed}_test{cap}.txt"),
            },
        }
        with open(publish_dir / "hparams.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        crash = publish_dir / "CRASH.txt"
        if crash.is_file():
            crash.unlink()

    print(f"Saved seed={args.seed} ablation={tag} dataset={dataset}", flush=True)
    return rows


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--max-val", type=int, default=800)
    ap.add_argument("--max-test", type=int, default=1200)
    ap.add_argument("--epochs-gnn", type=int, default=8)
    ap.add_argument("--epochs-seq", type=int, default=8)
    ap.add_argument("--batch-gnn", type=int, default=6)
    ap.add_argument("--batch-seq", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--lambda-proto", type=float, default=0.3)
    ap.add_argument("--lambda-view", type=float, default=0.2)
    ap.add_argument("--no-dual-view", action="store_true")
    ap.add_argument("--no-prototypes", action="store_true")
    ap.add_argument("--no-motifs", action="store_true")
    ap.add_argument("--no-fusion", action="store_true")
    ap.add_argument(
        "--ckpt-dir",
        type=str,
        default="",
        help="Directory for stage checkpoints (default: publish_dir or results_protocl/ckpt_seed{seed}_{ablation})",
    )
    ap.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore existing checkpoints and metrics.csv; train from scratch",
    )
    ap.add_argument(
        "--no-cache-graphs",
        action="store_true",
        help="Do not keep PyG graphs in a RAM LRU (disk CPG cache still used)",
    )
    ap.add_argument(
        "--cpg-cache-dir",
        type=str,
        default="",
        help="Directory for pickled tree-sitter CPGs (default: datasets/reveal_cpg_cache or primevul_cpg_cache)",
    )
    return ap


def main():
    args = build_argparser().parse_args()
    run_protocl_once(args)


if __name__ == "__main__":
    main()
