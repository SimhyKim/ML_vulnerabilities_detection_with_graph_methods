
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader as GraphDataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpg_vuln.graph_dataset import (
    CPGVulnDataset,
    EDGE_KIND_TO_ID,
    NODE_FLAG_DIM,
    NUM_TOKEN_BUCKETS,
    NUM_TYPE_BUCKETS,
)
from cpg_vuln.model import VulCPGGNN
from cpg_vuln.model_plus import VulCPGPlus
from vulprotocl.data import (
    DEFAULT_PRIMEVUL_DIR,
    default_reveal_dir,
    load_primevul_official,
    load_reveal,
    stratified_split,
    subsample,
    write_split_hashes,
)
from vulprotocl.metrics import FocalLoss, best_threshold, metrics_from_probs, set_seed

__all__ = [
    "DEFAULT_PRIMEVUL_DIR",
    "FocalLoss",
    "best_threshold",
    "load_primevul_official",
    "load_reveal",
    "metrics_from_probs",
    "set_seed",
    "stratified_split",
    "subsample",
    "write_split_hashes",
]


@dataclass
class RunResult:
    config: str
    dataset: str
    n_train: int
    n_val: int
    n_test: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    balanced_acc: float
    retrieval_alpha: float = 1.0


def make_model(name: str, device: torch.device):
    n_rel = max(EDGE_KIND_TO_ID.values()) + 1
    if name in {"baseline_cpg", "spg"}:
        return VulCPGGNN(
            num_type_buckets=NUM_TYPE_BUCKETS,
            num_token_buckets=NUM_TOKEN_BUCKETS,
            num_edge_types=n_rel,
            flag_dim=NODE_FLAG_DIM,
            hidden_dim=192,
            num_layers=3,
        ).to(device)
    return VulCPGPlus(
        num_type_buckets=NUM_TYPE_BUCKETS,
        num_token_buckets=NUM_TOKEN_BUCKETS,
        num_edge_types=n_rel,
        use_domain=True,
        node_flag_dim=NODE_FLAG_DIM,
        hidden_dim=192,
        num_layers=4,
    ).to(device)


def make_dataset(split, config: str) -> CPGVulnDataset:
    return CPGVulnDataset(
        split,
        use_slice=config != "baseline_cpg",
        slice_hops=2,
        use_domain_graph_feat=True,
    )


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        _, logit, _ = model(batch)
        logits.extend(logit.detach().cpu().numpy().tolist())
        labels.extend(batch.y.detach().cpu().numpy().tolist())
    return np.array(logits), np.array(labels)


def train_config(config, train_split, val_split, test_split, epochs, batch_size, lr, device, seed):
    set_seed(seed)
    model = make_model(config, device)
    train_ds, val_ds, test_ds = (
        make_dataset(train_split, config),
        make_dataset(val_split, config),
        make_dataset(test_split, config),
    )
    y_train = np.array(train_split["vul"])
    num_pos = max(int(y_train.sum()), 1)
    num_neg = max(len(y_train) - num_pos, 1)
    criterion = FocalLoss(torch.tensor([num_neg / num_pos], device=device, dtype=torch.float32), gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sample_weights = np.where(y_train == 1, float(num_neg) / num_pos, 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = GraphDataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = GraphDataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_ds, batch_size=batch_size, shuffle=False)

    best_state, best_val_f1, best_th = None, -1.0, 0.5
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            _, logits, _ = model(batch)
            loss = criterion(logits, batch.y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val_logits, val_y = collect_logits(model, val_loader, device)
        val_probs = 1 / (1 + np.exp(-val_logits))
        th, vf1 = best_threshold(val_y, val_probs)
        print(f"[{config}] epoch {ep+1}/{epochs} val_f1={vf1:.4f} th={th:.2f}", flush=True)
        if vf1 > best_val_f1:
            best_val_f1, best_th = vf1, th
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    te_logits, te_y = collect_logits(model, test_loader, device)
    te_probs = 1 / (1 + np.exp(-te_logits))
    m = metrics_from_probs(te_y, te_probs, best_th)
    extra = {k: v for k, v in m.items() if k in RunResult.__dataclass_fields__}
    return RunResult(
        config=config,
        dataset="",
        n_train=len(train_split),
        n_val=len(val_split),
        n_test=len(test_split),
        threshold=best_th,
        retrieval_alpha=1.0,
        **extra,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", nargs="+", default=["baseline_cpg", "spg_domain"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = load_reveal(default_reveal_dir())
    train, val, test = stratified_split(full, args.seed)
    train, val, test = (
        subsample(train, args.max_train, args.seed),
        subsample(val, args.max_val, args.seed + 1),
        subsample(test, args.max_test, args.seed + 2),
    )
    print(f"Device={device} train={len(train)} val={len(val)} test={len(test)}", flush=True)

    results: List[RunResult] = []
    for cfg in args.configs:
        r = train_config(cfg, train, val, test, args.epochs, args.batch_size, args.lr, device, args.seed)
        r.dataset = "reveal"
        results.append(r)
        print(asdict(r), flush=True)

    out_dir = ROOT / "results" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"results_reveal_seed{args.seed}.csv"
    fields = list(asdict(results[0]).keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"Saved {csv_path}", flush=True)


if __name__ == "__main__":
    main()
