
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset as HFDataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMEVUL_DIR = ROOT / "datasets" / "primevul"


def default_reveal_dir() -> Path:
    here = ROOT / "datasets" / "reveal"
    if (here / "vulnerables.json").is_file():
        return here
    sibling = ROOT.parent / "datasets" / "reveal"
    if (sibling / "vulnerables.json").is_file():
        return sibling
    raise FileNotFoundError(
        "Place ReVeal JSON in datasets/reveal/ (vulnerables.json + non-vulnerables.json). "
        "See datasets/README.md."
    )


def default_primevul_dir() -> Path:
    here = DEFAULT_PRIMEVUL_DIR
    if (here / "primevul_train.jsonl").is_file():
        return here
    alt = ROOT.parent / "experiments" / "dataset PrimeVul"
    if (alt / "primevul_train.jsonl").is_file():
        return alt
    return here


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_reveal(path: Path) -> HFDataset:
    with open(path / "vulnerables.json", "r", encoding="utf-8") as f:
        vul = json.load(f)
    with open(path / "non-vulnerables.json", "r", encoding="utf-8") as f:
        non = json.load(f)
    rows: List[Dict[str, Any]] = []
    for item, y in ((vul, 1), (non, 0)):
        for rec in item:
            code = rec.get("code") or rec.get("func") or rec.get("function")
            if not code:
                continue
            rows.append(
                {
                    "func_before": code,
                    "vul": y,
                    "hash": str(rec.get("hash", "")),
                    "project": str(rec.get("project", "")),
                }
            )
    return HFDataset.from_list(rows)


def load_primevul_jsonl(path: Path) -> HFDataset:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            func = o.get("func") or o.get("func_before") or ""
            if not func:
                continue
            rows.append(
                {
                    "func_before": func,
                    "vul": int(o.get("target", o.get("vul", 0))),
                    "hash": str(o.get("hash", o.get("idx", ""))),
                    "project": str(o.get("project", "")),
                    "idx": str(o.get("idx", "")),
                }
            )
    return HFDataset.from_list(rows)


def load_primevul_official(path: Path | None = None) -> Tuple[HFDataset, HFDataset, HFDataset]:
    d = Path(path) if path is not None else default_primevul_dir()
    return (
        load_primevul_jsonl(d / "primevul_train.jsonl"),
        load_primevul_jsonl(d / "primevul_valid.jsonl"),
        load_primevul_jsonl(d / "primevul_test.jsonl"),
    )


def write_split_hashes(split: HFDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    codes = list(split["func_before"])
    orig = list(split["hash"]) if "hash" in split.column_names else [""] * len(split)
    with open(path, "w", encoding="utf-8") as f:
        f.write("reveal_hash\tcode_sha256\n")
        for h, code in zip(orig, codes):
            digest = hashlib.sha256(str(code).encode("utf-8", errors="replace")).hexdigest()
            f.write(f"{h}\t{digest}\n")


def stratified_split(
    ds: HFDataset, seed: int, val_ratio: float = 0.1, test_ratio: float = 0.15
) -> Tuple[HFDataset, HFDataset, HFDataset]:
    labels = np.array(ds["vul"])
    idx = np.arange(len(ds))
    rng = np.random.RandomState(seed)

    def _split_class(c: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cidx = idx[labels == c]
        rng.shuffle(cidx)
        n = len(cidx)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        return cidx[:n_test], cidx[n_test : n_test + n_val], cidx[n_test + n_val :]

    parts = [_split_class(0), _split_class(1)]
    train_idx = np.concatenate([p[2] for p in parts])
    val_idx = np.concatenate([p[1] for p in parts])
    test_idx = np.concatenate([p[0] for p in parts])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return ds.select(train_idx.tolist()), ds.select(val_idx.tolist()), ds.select(test_idx.tolist())


def subsample(ds: HFDataset, n: int, seed: int) -> HFDataset:
    if n <= 0 or n >= len(ds):
        return ds
    rng = np.random.RandomState(seed)
    labels = np.array(ds["vul"])
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    n_pos = min(len(pos), max(50, int(n * (len(pos) / max(len(ds), 1)))))
    n_neg = min(len(neg), n - n_pos)
    if n_neg <= 0:
        n_neg = min(len(neg), n // 2)
        n_pos = min(len(pos), n - n_neg)
    if n < 100:
        n_pos = min(len(pos), max(1, int(round(n * (len(pos) / max(len(ds), 1))))))
        n_neg = min(len(neg), max(1, n - n_pos))
        if n_pos + n_neg > n:
            n_neg = max(1, n - n_pos)
            n_pos = n - n_neg
        n_pos = min(n_pos, len(pos))
        n_neg = min(n_neg, len(neg))
    choose = np.concatenate(
        [
            rng.choice(pos, size=n_pos, replace=False),
            rng.choice(neg, size=n_neg, replace=False),
        ]
    )
    rng.shuffle(choose)
    return ds.select(sorted(choose.tolist()))
