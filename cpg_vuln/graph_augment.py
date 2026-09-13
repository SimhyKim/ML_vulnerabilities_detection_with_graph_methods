
from __future__ import annotations

import random
from typing import Any, Dict, List

from cpg_vuln.slicer import extract_spg


def edge_dropout(cpg: Dict[str, Any], keep_prob: float = 0.85, rng: random.Random | None = None) -> Dict[str, Any]:
    rng = rng or random.Random()
    edges = [e for e in cpg.get("edges", []) if rng.random() < keep_prob]
    if not edges and cpg.get("nodes"):
        edges = [{"src": 0, "dst": 0, "kind": "UNKNOWN"}]
    out = dict(cpg)
    out["edges"] = edges
    return out


def dual_vulnerability_views(
    full_cpg: Dict[str, Any],
    rng: random.Random | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    rng = rng or random.Random()
    view_a = extract_spg(full_cpg, hops=2)
    view_b = extract_spg(full_cpg, hops=3)
    view_b = edge_dropout(view_b, keep_prob=0.9, rng=rng)
    return view_a, view_b
