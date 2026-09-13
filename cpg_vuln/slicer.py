
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Set

from cpg_vuln.domain_knowledge import ALL_SECURITY_APIS


SEED_NODE_TYPES: Set[str] = {
    "call_expression",
    "subscript_expression",
    "pointer_expression",
    "field_expression",
    "assignment_expression",
    "init_declarator",
    "return_statement",
    "if_statement",
    "for_statement",
    "while_statement",
}


def _normalize_name(text: str) -> str:
    return (text or "").strip()


def find_seed_ids(cpg: Dict[str, Any]) -> Set[int]:
    seeds: Set[int] = set()
    for node in cpg.get("nodes", []):
        nid = int(node["id"])
        ntype = str(node.get("type", ""))
        code = _normalize_name(str(node.get("code") or ""))
        name = code.split("(")[0].strip() if "(" in code else code

        if ntype in SEED_NODE_TYPES:
            seeds.add(nid)
        if name in ALL_SECURITY_APIS or code in ALL_SECURITY_APIS:
            seeds.add(nid)
        if "->" in code or (ntype == "identifier" and any(ch in code for ch in ("*", "&"))):
            seeds.add(nid)
        if ntype == "identifier" and name in ALL_SECURITY_APIS:
            seeds.add(nid)
    return seeds


def extract_spg(cpg: Dict[str, Any], hops: int = 2) -> Dict[str, Any]:
    nodes = cpg.get("nodes", [])
    edges = cpg.get("edges", [])
    if not nodes:
        return cpg

    seeds = find_seed_ids(cpg)
    if not seeds:
        return cpg

    adj: Dict[int, List[int]] = defaultdict(list)
    for e in edges:
        s, d = int(e["src"]), int(e["dst"])
        adj[s].append(d)
        adj[d].append(s)

    keep: Set[int] = set()
    q: deque = deque()
    for s in seeds:
        keep.add(s)
        q.append((s, 0))

    while q:
        cur, dist = q.popleft()
        if dist >= hops:
            continue
        for nb in adj.get(cur, []):
            if nb not in keep:
                keep.add(nb)
                q.append((nb, dist + 1))

    for node in nodes:
        if str(node.get("type", "")) in {"function_definition", "translation_unit"}:
            keep.add(int(node["id"]))

    id_map = {old: i for i, old in enumerate(sorted(keep))}
    new_nodes = []
    for node in nodes:
        oid = int(node["id"])
        if oid not in id_map:
            continue
        nn = dict(node)
        nn["id"] = id_map[oid]
        new_nodes.append(nn)

    new_edges = []
    for e in edges:
        s, d = int(e["src"]), int(e["dst"])
        if s in id_map and d in id_map:
            new_edges.append(
                {"src": id_map[s], "dst": id_map[d], "kind": e.get("kind", "UNKNOWN")}
            )

    if not new_nodes:
        return cpg
    if not new_edges:
        new_edges = [{"src": 0, "dst": 0, "kind": "UNKNOWN"}]

    return {"nodes": new_nodes, "edges": new_edges, "n_seeds": len(seeds), "n_keep": len(keep)}
