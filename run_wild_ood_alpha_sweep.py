
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vulprotocl.metrics import best_threshold, metrics_from_probs
from sklearn.metrics import average_precision_score

OOD_DIR = ROOT / "results" / "wild_ood"
PUBLISH = ROOT / "results"
OUT = OOD_DIR / "alpha_sweep"
ALPHAS = np.linspace(0.0, 1.0, 21)


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_ood(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = OOD_DIR / f"seed_{seed}" / "predictions.jsonl"
    y, pg, ps = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            y.append(int(rec["y"]))
            pg.append(float(rec["p_gnn"]))
            ps.append(float(rec.get("p_seq", rec["p_fusion"])))
    return np.array(y), np.array(pg), np.array(ps)


def load_reveal_val(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = PUBLISH / f"seed_{seed}" / "ckpt_scores.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return (
        np.asarray(blob["va_y"]),
        np.asarray(blob["va_pg"]),
        np.asarray(blob["va_ps"]),
    )


def blend(alpha: float, p_seq: np.ndarray, p_gnn: np.ndarray) -> np.ndarray:
    return alpha * p_seq + (1.0 - alpha) * p_gnn


def pick_on_val(va_y, va_pg, va_ps, policy: str) -> tuple[float, float, float]:
    if policy == "gnn_only":
        th, f1 = best_threshold(va_y, va_pg)
        return 0.0, float(th), float(f1)
    if policy == "seq_only":
        th, f1 = best_threshold(va_y, va_ps)
        return 1.0, float(th), float(f1)

    best = {"score": -1.0, "alpha": 0.0, "th": 0.5}
    cap = 1.0
    if policy == "val_f1_graph_cap":
        cap = 0.30
        policy_score = "f1"
    elif policy == "val_f1":
        policy_score = "f1"
    elif policy == "val_pr_auc":
        policy_score = "pr_auc"
    else:
        raise ValueError(policy)

    for alpha in ALPHAS:
        if alpha > cap + 1e-9:
            continue
        p = blend(alpha, va_ps, va_pg)
        if policy_score == "pr_auc":
            score = float(average_precision_score(va_y, p))
            th, _ = best_threshold(va_y, p)
        else:
            th, score = best_threshold(va_y, p)
            score = float(score)
        if score > best["score"]:
            best = {"score": score, "alpha": float(alpha), "th": float(th)}
    return best["alpha"], best["th"], best["score"]


def ood_oracle_grid(y, pg, ps) -> list[dict]:
    rows = []
    for alpha in ALPHAS:
        p = blend(float(alpha), ps, pg)
        th, f1 = best_threshold(y, p)
        rec = metrics_from_probs(y, p, th)
        rec.update(
            {
                "alpha": float(alpha),
                "threshold_oracle": float(th),
                "oracle_f1_at_best_th": float(f1),
                "kind": "ood_oracle_analysis_only",
            }
        )
        rows.append(rec)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    OUT.mkdir(parents=True, exist_ok=True)

    policies = ("val_f1", "val_pr_auc", "val_f1_graph_cap", "gnn_only", "seq_only")
    frozen_rows: list[dict] = []
    oracle_rows: list[dict] = []

    for seed in seeds:
        y, pg, ps = load_ood(seed)
        va_y, va_pg, va_ps = load_reveal_val(seed)
        print(f"\n===== seed {seed}  OOD n={len(y)} pos={int(y.sum())} =====", flush=True)
        for policy in policies:
            alpha, th, val_score = pick_on_val(va_y, va_pg, va_ps, policy)
            p = blend(alpha, ps, pg)
            rec = metrics_from_probs(y, p, th)
            rec.update(
                {
                    "seed": seed,
                    "policy": policy,
                    "alpha": alpha,
                    "threshold": th,
                    "reveal_val_score": val_score,
                    "n_ood": int(len(y)),
                    "n_ood_pos": int(y.sum()),
                    "selection": "ReVeal validation only; frozen on wild OOD",
                }
            )
            frozen_rows.append(rec)
            print(
                f"  {policy:18s} alpha={alpha:.2f} th={th:.3f}  "
                f"OOD F1={rec['f1']:.3f} PR-AUC={rec['pr_auc']:.3f} "
                f"R={rec['recall']:.3f} P={rec['precision']:.3f}",
                flush=True,
            )
        for rec in ood_oracle_grid(y, pg, ps):
            rec["seed"] = seed
            oracle_rows.append(rec)

    def write_csv(path: Path, rows: list[dict]) -> None:
        keys: list[str] = []
        seen: set[str] = set()
        for rec in rows:
            for k in rec:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    write_csv(OUT / "frozen_policies.csv", frozen_rows)
    write_csv(OUT / "ood_oracle_alpha_grid.csv", oracle_rows)

    summary = []
    for policy in policies:
        sub = [r for r in frozen_rows if r["policy"] == policy]
        rec: dict = {"policy": policy, "n_seeds": len(sub)}
        rec["alpha_mean"] = float(np.mean([r["alpha"] for r in sub]))
        for key in ("f1", "pr_auc", "roc_auc", "precision", "recall"):
            vals = [float(r[key]) for r in sub]
            rec[f"{key}_mean"] = float(np.mean(vals))
            rec[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary.append(rec)
        print(
            f"MEAN {policy:18s} alpha={rec['alpha_mean']:.2f}  "
            f"F1={rec['f1_mean']:.3f}±{rec['f1_std']:.3f}  "
            f"PR-AUC={rec['pr_auc_mean']:.3f}±{rec['pr_auc_std']:.3f}  "
            f"R={rec['recall_mean']:.3f}±{rec['recall_std']:.3f}",
            flush=True,
        )
    write_csv(OUT / "frozen_policies_mean.csv", summary)

    peek = []
    for seed in seeds:
        sub = [r for r in oracle_rows if r["seed"] == seed]
        best = max(sub, key=lambda r: r["f1"])
        peek.append(best)
        print(
            f"ORACLE seed={seed} alpha={best['alpha']:.2f} OOD F1={best['f1']:.3f} "
            f"(peeked at probe labels — analysis only)",
            flush=True,
        )
    write_csv(OUT / "ood_oracle_best.csv", peek)
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
