
from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vulprotocl.data import load_reveal, stratified_split, subsample, write_split_hashes, default_reveal_dir
from run_protocl import ablation_tag, build_argparser, run_protocl_once
from cpg_vuln.progress import fallen, progress

PUBLISH = ROOT / "results"
NUMERIC_KEYS = (
    "f1",
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "prec_at_rec_0.2",
    "prec_at_rec_0.4",
    "prec_at_rec_0.6",
    "best_epoch",
    "alpha",
    "threshold",
)


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def prepare_splits(seeds: list[int], max_train: int, max_val: int, max_test: int) -> None:
    full = load_reveal(default_reveal_dir())
    split_dir = PUBLISH / "splits"
    for seed in seeds:
        train, val, test = stratified_split(full, seed)
        train = subsample(train, max_train, seed)
        val = subsample(val, max_val, seed + 1)
        test = subsample(test, max_test, seed + 2)
        write_split_hashes(train, split_dir / f"reveal_seed{seed}_train.txt")
        write_split_hashes(val, split_dir / f"reveal_seed{seed}_val.txt")
        write_split_hashes(test, split_dir / f"reveal_seed{seed}_test.txt")
        meta = {
            "seed": seed,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_pos_train": int(sum(train["vul"])),
            "n_pos_val": int(sum(val["vul"])),
            "n_pos_test": int(sum(test["vul"])),
            "max_train": max_train,
            "max_val": max_val,
            "max_test": max_test,
        }
        seed_dir = PUBLISH / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with open(seed_dir / "split_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"prepared seed={seed} {meta}", flush=True)


def aggregate_seed_csvs(publish: Path = PUBLISH, out_name: str = "reveal_full_seeds.csv") -> Path | None:
    rows = []
    for metrics in sorted(publish.glob("seed_*/metrics.csv")):
        with open(metrics, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        print("No seed_*/metrics.csv yet — skip aggregate.", flush=True)
        return None

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["config"]].append(r)

    out_rows = []
    for config, items in grouped.items():
        rec: dict = {"config": config, "n_seeds": len(items)}
        for key in NUMERIC_KEYS:
            vals = []
            for it in items:
                raw = it.get(key, "")
                if raw in ("", None):
                    continue
                try:
                    vals.append(float(raw))
                except ValueError:
                    continue
            if not vals:
                continue
            rec[f"{key}_mean"] = float(np.mean(vals))
            rec[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            rec[f"{key}_cell"] = (
                f"{rec[f'{key}_mean']:.3f} ± {rec[f'{key}_std']:.3f}"
                if len(vals) > 1
                else f"{rec[f'{key}_mean']:.3f}"
            )
        out_rows.append(rec)

    out = publish / out_name
    keys = sorted({k for r in out_rows for k in r.keys()})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"Wrote {out}", flush=True)
    return out


def write_readme() -> None:
    PUBLISH.mkdir(parents=True, exist_ok=True)
    text = """VulProtoCL multi-seed outputs
================================
seed_<N>/metrics.csv     per-seed test metrics (protocl_gnn, bilstm_ref, protocl_late_fusion)
seed_<N>/hparams.json    argv, best_epoch, split file paths
seed_<N>/protocl_best.pt  final fused checkpoint
seed_<N>/ckpt_gnn.pt     GNN weights (resume skips GNN train)
seed_<N>/ckpt_bilstm.pt  BiLSTM weights (resume skips BiLSTM train)
seed_<N>/ckpt_scores.pt  val/test probabilities (resume skips forwards)
splits/reveal_seed<N>_*.txt  fingerprints (reveal_hash + code SHA-256)
  no suffix = full ReVeal; _capXXXX = subsampled debug run
reveal_full_seeds.csv    mean±std across seeds that currently have matching metrics.csv

Resume: re-run the same command. Compatible checkpoints in seed_<N>/ are loaded;
incompatible files (e.g. a 16-sample smoke test vs full ReVeal) are ignored.
--force-retrain trains from scratch.
"""
    (PUBLISH / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    base = build_argparser()
    base.add_argument("--seeds", type=str, default="42,43,44")
    base.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write split hash files only; do not train",
    )
    faulthandler.enable()
    args = base.parse_args()
    seeds = parse_seeds(args.seeds)
    write_readme()
    progress(f"driver start seeds={seeds} max_train={args.max_train} epochs_gnn={args.epochs_gnn}", stage="driver")

    if args.prepare_only:
        prepare_splits(seeds, args.max_train, args.max_val, args.max_test)
        print(f"Split files under {PUBLISH / 'splits'}", flush=True)
        return

    tag = ablation_tag(args)
    if tag == "full":
        run_root = PUBLISH
        out_name = "reveal_full_seeds.csv"
    else:
        run_root = PUBLISH / "ablations" / tag
        out_name = "mean_std.csv"
        progress(f"ablation={tag} → {run_root} (will not overwrite seed_*/ full-model files)")

    for seed in seeds:
        print(f"\n===== VulProtoCL seed {seed} ablation={tag} =====", flush=True)
        progress(f"begin seed {seed} ablation={tag}", stage=f"seed_{seed}")
        args.seed = seed
        seed_dir = run_root / f"seed_{seed}"
        try:
            run_protocl_once(args, publish_dir=seed_dir)
        except Exception as err:
            fallen(f"seed={seed}", err)
            crash = seed_dir / "CRASH.txt"
            seed_dir.mkdir(parents=True, exist_ok=True)
            crash.write_text(
                f"FALLEN at seed={seed}\n{type(err).__name__}: {err}\n\n{traceback.format_exc()}",
                encoding="utf-8",
            )
            print(f"Wrote {crash} — stopping remaining seeds.", flush=True)
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        aggregate_seed_csvs(publish=run_root, out_name=out_name)
        progress(f"seed {seed} finished OK", stage=f"seed_{seed}_done")

    summary = run_root / out_name
    print(
        f"\nDone. Open {summary} and {run_root / f'seed_{seeds[0]}' / 'metrics.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
