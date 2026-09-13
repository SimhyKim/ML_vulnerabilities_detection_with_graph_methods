
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Set

from cpg_vuln.domain_knowledge import ALLOC_APIS, INJECTION_SINKS, MEMORY_SINKS, SOURCES

MOTIF_DIM = 16


def _ids_by_type(cpg: Dict[str, Any]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = defaultdict(list)
    for n in cpg.get("nodes", []):
        out[str(n.get("type", ""))].append(int(n["id"]))
    return out


def _adj(cpg: Dict[str, Any], kinds: Set[str] | None = None) -> Dict[int, List[int]]:
    a: Dict[int, List[int]] = defaultdict(list)
    for e in cpg.get("edges", []):
        if kinds is not None and str(e.get("kind", "")) not in kinds:
            continue
        a[int(e["src"])].append(int(e["dst"]))
    return a


def _reachable(adj: Dict[int, List[int]], starts: Set[int], max_hops: int = 6) -> Set[int]:
    seen: Set[int] = set()
    q = deque([(s, 0) for s in starts])
    while q:
        cur, d = q.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        if d >= max_hops:
            continue
        for nb in adj.get(cur, []):
            q.append((nb, d + 1))
    return seen


def _api_nodes(cpg: Dict[str, Any], apis: Set[str]) -> Set[int]:
    hits = set()
    for n in cpg.get("nodes", []):
        code = str(n.get("code") or "").strip()
        name = code.split("(")[0].strip() if "(" in code else code
        if name in apis or code in apis:
            hits.add(int(n["id"]))
    return hits


def extract_motifs(cpg: Dict[str, Any]) -> List[float]:
    nodes = cpg.get("nodes", [])
    n = max(len(nodes), 1)
    by_type = _ids_by_type(cpg)
    dfg = _adj(cpg, {"DFG", "DFG_REV", "CALL_ARG", "CALL_ARG_REV", "CALL_CALLEE", "CALL_CALLEE_REV"})
    ctrl = _adj(cpg, {"CTRL_DEP", "CTRL_DEP_REV"})

    mem = _api_nodes(cpg, MEMORY_SINKS)
    inj = _api_nodes(cpg, INJECTION_SINKS)
    alloc = _api_nodes(cpg, ALLOC_APIS)
    src = _api_nodes(cpg, SOURCES)
    frees = _api_nodes(cpg, {"free", "delete"})

    checks = set(by_type.get("if_statement", []) + by_type.get("conditional_expression", []))
    arrays = set(by_type.get("subscript_expression", []))
    ptrs = {
        int(nd["id"])
        for nd in nodes
        if ("->" in str(nd.get("code") or "") or str(nd.get("type", "")) == "pointer_expression")
    }

    src_reach = _reachable(dfg, src, 8)
    m_src_to_sink = 1.0 if (src_reach & mem) else 0.0

    sink_ctrl = _reachable(ctrl, mem, 3)
    m_sink_no_check = 1.0 if mem and not (sink_ctrl & checks) else 0.0

    free_reach = _reachable(dfg, frees, 6)
    m_uaf_proxy = 1.0 if frees and len(free_reach) > len(frees) + 2 else 0.0

    m_inj_taint = 1.0 if (src_reach & inj) else 0.0

    assigns = set(by_type.get("assignment_expression", []))
    m_array_write = 1.0 if (arrays & _reachable(dfg, assigns, 2)) or (arrays and assigns) else 0.0
    m_ptr_write = min(1.0, len(ptrs) / 5.0)

    dens_mem = len(mem) / n
    dens_inj = len(inj) / n
    dens_alloc = len(alloc) / n
    dens_src = len(src) / n
    dens_ctrl = len(checks) / n
    dens_call = len(by_type.get("call_expression", [])) / n
    dens_ret = len(by_type.get("return_statement", [])) / n

    m_alloc_no_free = 1.0 if alloc and not frees else 0.0

    m_goto = 1.0 if by_type.get("goto_statement") else 0.0

    return [
        m_src_to_sink,
        m_sink_no_check,
        m_uaf_proxy,
        m_inj_taint,
        m_array_write,
        m_ptr_write,
        dens_mem,
        dens_inj,
        dens_alloc,
        dens_src,
        dens_ctrl,
        dens_call,
        dens_ret,
        m_alloc_no_free,
        m_goto,
        float(min(len(nodes), 500) / 500.0),
    ]


assert len(extract_motifs({"nodes": [], "edges": []})) == MOTIF_DIM
