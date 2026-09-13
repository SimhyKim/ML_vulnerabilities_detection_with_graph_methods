
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpg_vuln.graph_dataset import CPGVulnDataset
from cpg_vuln.progress import progress
from vulprotocl.seq import BiLSTM, SeqDS, predict_seq
from vulprotocl.metrics import metrics_from_probs, set_seed, sigmoid
from vulprotocl.data import subsample
from run_protocl import (
    ablation_tag,
    build_argparser,
    build_protocl_model,
    predict_protocl,
    run_protocl_once,
    write_metrics_csv,
    write_metrics_json,
)

WILD_DIR = ROOT / "datasets" / "wild_ood"
if not (WILD_DIR / "vulnerables.json").is_file():
    _alt = ROOT.parent / "datasets" / "wild_ood"
    if (_alt / "vulnerables.json").is_file():
        WILD_DIR = _alt
OUT_DIR = ROOT / "results" / "wild_ood"
CPG_CACHE = ROOT / "datasets" / "wild_ood_cpg_cache"
NUMERIC_KEYS = (
    "f1",
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "prec_at_rec_0.2",
    "prec_at_rec_0.4",
    "prec_at_rec_0.6",
    "alpha",
    "threshold",
    "best_epoch",
)


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_wild_ood(path: Path) -> HFDataset:
    vul = json.loads((path / "vulnerables.json").read_text(encoding="utf-8"))
    non = json.loads((path / "non-vulnerables.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item, default_y in ((vul, 1), (non, 0)):
        for rec in item:
            code = rec.get("code") or rec.get("func") or rec.get("function") or ""
            if not code:
                continue
            cwe = rec.get("cwe") or []
            if isinstance(cwe, str):
                cwe = [cwe] if cwe else []
            rows.append(
                {
                    "func_before": str(code),
                    "vul": float(rec.get("label", rec.get("vul", default_y))),
                    "hash": str(rec.get("hash", "")),
                    "project": str(rec.get("project", "")),
                    "cve": str(rec.get("cve", "")),
                    "cwe": ",".join(str(c) for c in cwe),
                    "file": str(rec.get("file", "")),
                    "function": str(rec.get("function", "")),
                    "commit_url": str(rec.get("commit_url", "")),
                    "nvd_url": str(rec.get("nvd_url", "")),
                    "source": str(rec.get("source", "")),
                    "published": str(rec.get("published", "")),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No functions in {path}")
    return HFDataset.from_list(rows)


def _pt_candidates(ckpt: Path, seed: int) -> list[Path]:
    names = [
        ckpt,
        ckpt / "protocl_best.pt",
        ckpt / f"protocl_reveal_full_seed{seed}_best.pt",
        ROOT / "results" / f"seed_{seed}" / "protocl_best.pt",
    ]
    out: list[Path] = []
    seen: set[Path] = set()
    for p in names:
        rp = p.resolve() if p.exists() else p
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def _fusion_from_metrics(folder: Path) -> tuple[float, float, int]:
    alpha, th, best_epoch = 0.5, 0.5, 0
    metrics = folder / "metrics.json"
    if not metrics.is_file():
        metrics = folder / "metrics.csv"
        if metrics.is_file():
            with metrics.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("config") == "protocl_late_fusion":
                        alpha = float(row.get("alpha") or alpha)
                        th = float(row.get("threshold") or th)
                        best_epoch = int(float(row.get("best_epoch") or 0))
                        return alpha, th, best_epoch
        return alpha, th, best_epoch
    rows = json.loads(metrics.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("config") == "protocl_late_fusion":
            alpha = float(row.get("alpha", alpha))
            th = float(row.get("threshold", th))
            best_epoch = int(row.get("best_epoch") or 0)
    return alpha, th, best_epoch


def load_reveal_ckpt(ckpt: Path, seed: int, device: torch.device, no_fusion: bool):
    for path in _pt_candidates(ckpt, seed):
        if not path.is_file():
            continue
        progress(f"loading checkpoint {path}")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "protocl" not in blob:
            continue
        gnn = build_protocl_model(device)
        gnn.load_state_dict(blob["protocl"])
        gnn.eval()
        alpha = float(blob.get("alpha", 0.5))
        th = float(blob.get("threshold", 0.5))
        best_epoch = int(blob.get("best_epoch") or 0)
        folder = path.parent
        m_alpha, m_th, m_ep = _fusion_from_metrics(folder)
        if "alpha" not in blob:
            alpha = m_alpha
        if "threshold" not in blob:
            th = m_th
        if not best_epoch:
            best_epoch = m_ep
        bilstm = None
        vocab = None
        if not no_fusion:
            if "bilstm" in blob and "vocab" in blob:
                vocab = blob["vocab"]
                bilstm = BiLSTM(len(vocab)).to(device)
                bilstm.load_state_dict(blob["bilstm"])
                bilstm.eval()
            else:
                bi_path = folder / "ckpt_bilstm.pt"
                if bi_path.is_file():
                    bi = torch.load(bi_path, map_location="cpu", weights_only=False)
                    vocab = bi["vocab"]
                    bilstm = BiLSTM(len(vocab)).to(device)
                    bilstm.load_state_dict(bi["bilstm"])
                    bilstm.eval()
        return gnn, bilstm, vocab, alpha, th, best_epoch, path
    return None


def train_reveal_if_needed(args) -> Path:
    tag = ablation_tag(args)
    if int(args.max_train) <= 0:
        publish = ROOT / "results" / f"seed_{args.seed}"
        if tag != "full":
            publish = ROOT / "results" / "ablations" / tag / f"seed_{args.seed}"
    else:
        publish = (
            OUT_DIR
            / f"reveal_train_cap{args.max_train}_seed{args.seed}_{tag}"
        )
    best = publish / "protocl_best.pt"
    if best.is_file() and not args.force_retrain:
        progress(f"ReVeal checkpoint already present: {best}")
        return publish
    progress(f"training VulProtoCL on ReVeal -> {publish}", stage="train_reveal")
    run_protocl_once(args, publish_dir=publish)
    return publish


def group_rows(probe: HFDataset, y: np.ndarray, probs: np.ndarray, th: float, key: str) -> list[dict]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for i in range(len(probe)):
        raw = str(probe[i].get(key) or "")
        parts = [p for p in raw.split(",") if p] if key == "cwe" else [raw or "unknown"]
        for part in parts or ["unknown"]:
            buckets[part].append(i)
    out = []
    for name, idxs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        if len(idxs) < 3:
            continue
        yy = y[idxs]
        pp = probs[idxs]
        rec = {
            "group": key,
            "name": name,
            "n": int(len(idxs)),
            "n_pos": int(yy.sum()),
            **metrics_from_probs(yy, pp, th),
        }
        out.append(rec)
    return out


def evaluate_probe(
    probe: HFDataset,
    gnn,
    bilstm,
    vocab,
    alpha: float,
    threshold: float,
    device: torch.device,
    batch_gnn: int,
    batch_seq: int,
    max_len: int,
    use_motifs: bool,
    cache_graphs: bool,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    gnn_ds = CPGVulnDataset(
        probe,
        use_slice=True,
        slice_hops=2,
        use_domain_graph_feat=True,
        use_motifs=use_motifs,
        cache_graphs=cache_graphs,
        cache_dir=CPG_CACHE,
    )
    gnn_loader = GraphDataLoader(gnn_ds, batch_size=batch_gnn, shuffle=False)
    progress(f"GNN forward on wild OOD n={len(probe)}", stage="ood_gnn")
    g_logits, y = predict_protocl(gnn, gnn_loader, device)
    p_g = sigmoid(g_logits)

    p_s = None
    if bilstm is not None and vocab is not None:
        seq_ds = SeqDS(list(probe["func_before"]), y, vocab, max_len)
        seq_loader = DataLoader(seq_ds, batch_size=batch_seq, shuffle=False)
        progress("BiLSTM forward on wild OOD", stage="ood_seq")
        s_logits, _ = predict_seq(bilstm, seq_loader, device)
        p_s = sigmoid(s_logits)
        p_f = alpha * p_s + (1.0 - alpha) * p_g
    else:
        p_f = p_g
    return (
        [
            {"config": "protocl_gnn", **metrics_from_probs(y, p_g, threshold)},
            *(
                [{"config": "bilstm_ref", **metrics_from_probs(y, p_s, threshold)}]
                if p_s is not None
                else []
            ),
            {"config": "protocl_late_fusion", **metrics_from_probs(y, p_f, threshold)},
        ],
        y,
        p_g,
        p_f if p_s is None else p_f,
        p_s,
    )


def write_predictions(
    path: Path,
    probe: HFDataset,
    y: np.ndarray,
    p_g: np.ndarray,
    p_f: np.ndarray,
    p_s: np.ndarray | None,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(len(probe)):
            row = probe[i]
            rec = {
                "idx": i,
                "y": int(y[i]),
                "p_gnn": float(p_g[i]),
                "p_fusion": float(p_f[i]),
                "pred": int(p_f[i] >= threshold),
                "project": row.get("project", ""),
                "cve": row.get("cve", ""),
                "cwe": row.get("cwe", ""),
                "file": row.get("file", ""),
                "function": row.get("function", ""),
                "commit_url": row.get("commit_url", ""),
                "nvd_url": row.get("nvd_url", ""),
            }
            if p_s is not None:
                rec["p_seq"] = float(p_s[i])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def aggregate_seeds(out_dir: Path) -> Path | None:
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for metrics in sorted(out_dir.glob("seed_*/metrics.csv")):
        with metrics.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("split") and row["split"] != "wild_ood":
                    continue
                by_cfg[str(row.get("config"))].append(row)
    if not by_cfg:
        return None
    summary = []
    for cfg, rows in by_cfg.items():
        rec: dict[str, Any] = {
            "config": cfg,
            "n_seeds": len(rows),
            "n_ood": rows[0].get("n_ood", ""),
            "n_ood_pos": rows[0].get("n_ood_pos", ""),
        }
        for key in NUMERIC_KEYS:
            vals = []
            for row in rows:
                raw = row.get(key)
                if raw in (None, ""):
                    continue
                try:
                    vals.append(float(raw))
                except ValueError:
                    continue
            if not vals:
                continue
            rec[f"{key}_mean"] = float(np.mean(vals))
            rec[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary.append(rec)
    dest = out_dir / "wild_ood_seeds.csv"
    write_metrics_csv(dest, summary)
    return dest


def run_one_seed(args, probe: HFDataset) -> list[dict]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} seed={args.seed}", flush=True)
    ckpt_hint = Path(args.ckpt) if args.ckpt else ROOT / "results" / f"seed_{args.seed}"
    loaded = None if args.force_retrain else load_reveal_ckpt(
        ckpt_hint, args.seed, device, args.no_fusion
    )
    if loaded is None:
        if not args.train_reveal:
            raise FileNotFoundError(
                "No ReVeal VulProtoCL checkpoint found. Pass --ckpt PATH or "
                "re-train then evaluate with --train-reveal "
                "(full data: --max-train 0 --max-val 0 --max-test 0)."
            )
        reveal_dir = train_reveal_if_needed(args)
        loaded = load_reveal_ckpt(reveal_dir, args.seed, device, args.no_fusion)
        if loaded is None:
            raise RuntimeError(f"Training finished but no loadable checkpoint in {reveal_dir}")
    gnn, bilstm, vocab, alpha, threshold, best_epoch, ckpt_path = loaded
    if args.no_fusion:
        bilstm, vocab = None, None
    progress(
        f"frozen ReVeal alpha={alpha:.3f} threshold={threshold:.3f} "
        f"best_epoch={best_epoch} ckpt={ckpt_path}"
    )

    rows, y, p_g, p_f, p_s = evaluate_probe(
        probe,
        gnn,
        bilstm,
        vocab,
        alpha,
        threshold,
        device,
        args.batch_gnn,
        args.batch_seq,
        args.max_len,
        use_motifs=not args.no_motifs,
        cache_graphs=not args.no_cache_graphs,
    )
    seed_dir = OUT_DIR / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    meta_rows = []
    for rec in rows:
        rec.update(
            {
                "split": "wild_ood",
                "seed": args.seed,
                "alpha": alpha,
                "threshold": threshold,
                "best_epoch": best_epoch,
                "n_ood": int(len(y)),
                "n_ood_pos": int(y.sum()),
                "ckpt": str(ckpt_path),
                "dataset": "wild_ood",
            }
        )
        meta_rows.append(rec)
        print(rec, flush=True)

    groups = group_rows(probe, y, p_f, threshold, "project")
    groups += group_rows(probe, y, p_f, threshold, "cwe")
    write_metrics_csv(seed_dir / "metrics.csv", meta_rows)
    write_metrics_json(seed_dir / "metrics.json", meta_rows)
    if groups:
        write_metrics_csv(seed_dir / "metrics_by_group.csv", groups)
    write_predictions(seed_dir / "predictions.jsonl", probe, y, p_g, p_f, p_s, threshold)
    hparams = {
        "seed": args.seed,
        "ckpt": str(ckpt_path),
        "alpha": alpha,
        "threshold": threshold,
        "best_epoch": best_epoch,
        "n_ood": int(len(y)),
        "n_ood_pos": int(y.sum()),
        "protocol": "train/load ReVeal; freeze alpha+threshold; eval wild_ood once",
        "args": vars(args),
    }
    (seed_dir / "hparams.json").write_text(json.dumps(hparams, indent=2, default=str), encoding="utf-8")
    return meta_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=str, default=str(WILD_DIR))
    ap.add_argument("--ckpt", type=str, default="", help="protocl_best.pt or a seed folder")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default="", help="Override --seed, e.g. 42,43,44")
    ap.add_argument(
        "--train-reveal",
        action="store_true",
        help="Train VulProtoCL on ReVeal if no checkpoint is found",
    )
    ap.add_argument("--max-ood", type=int, default=0, help="Subsample the probe (0 = all 1110)")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--epochs-gnn", type=int, default=8)
    ap.add_argument("--epochs-seq", type=int, default=8)
    ap.add_argument("--batch-gnn", type=int, default=6)
    ap.add_argument("--batch-seq", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--lambda-proto", type=float, default=0.25)
    ap.add_argument("--lambda-view", type=float, default=0.15)
    ap.add_argument("--no-dual-view", action="store_true")
    ap.add_argument("--no-prototypes", action="store_true")
    ap.add_argument("--no-motifs", action="store_true")
    ap.add_argument("--no-fusion", action="store_true")
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--no-cache-graphs", action="store_true")
    ap.add_argument("--ckpt-dir", type=str, default="")
    args = ap.parse_args()
    args.cpg_cache_dir = str(ROOT / "datasets" / "reveal_cpg_cache")
    args.dataset = "reveal"

    seeds = parse_seeds(args.seeds) if args.seeds else [args.seed]
    data_dir = Path(args.data)
    progress(f"loading wild OOD from {data_dir}")
    probe = load_wild_ood(data_dir)
    n_pos = int(sum(probe["vul"]))
    progress(f"wild OOD n={len(probe)} pos={n_pos} neg={len(probe) - n_pos}")
    if args.max_ood > 0:
        probe = subsample(probe, args.max_ood, seeds[0])
        progress(f"subsampled max_ood={args.max_ood} n={len(probe)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CPG_CACHE.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        args.seed = seed
        print(f"\n===== wild OOD seed {seed} =====", flush=True)
        run_one_seed(args, probe)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = aggregate_seeds(OUT_DIR)
    print(f"\nDone. Metrics: {OUT_DIR / f'seed_{seeds[0]}' / 'metrics.csv'}", flush=True)
    if summary is not None:
        print(f"Seed summary: {summary}", flush=True)


if __name__ == "__main__":
    _ = build_argparser
    main()
