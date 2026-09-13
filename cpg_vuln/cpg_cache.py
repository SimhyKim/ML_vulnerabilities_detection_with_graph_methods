
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "datasets" / "reveal_cpg_cache"

_cache_hits = 0
_cache_miss = 0


def cache_stats() -> str:
    return f"cpg_cache hits={_cache_hits} miss={_cache_miss}"


def cache_key(code: str, func_hash: str = "") -> str:
    return hashlib.sha256((code or "").encode("utf-8", errors="replace")).hexdigest()


def load_or_build_cpg(
    code: str,
    func_hash: str,
    extractor: Any,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict:
    global _cache_hits, _cache_miss
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(code, func_hash)
    path = cache_dir / f"{key}.pkl"
    if path.exists():
        _cache_hits += 1
        if _cache_hits in (1, 10, 50) or _cache_hits % 200 == 0:
            from cpg_vuln.progress import progress

            progress(f"CPG disk HIT #{_cache_hits} ({cache_stats()})")
        with open(path, "rb") as f:
            return pickle.load(f)
    _cache_miss += 1
    if _cache_miss in (1, 5, 20) or _cache_miss % 50 == 0:
        from cpg_vuln.progress import progress

        progress(f"CPG disk MISS #{_cache_miss} — parsing tree-sitter ({cache_stats()})")
    cpg = extractor.build_cpg_from_code(code)
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cpg, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return cpg
