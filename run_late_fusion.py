
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
from cpg_vuln.model_plus import VulCPGPlus
from vulprotocl.seq import BiLSTM, SeqDS, predict_seq, train_bilstm
from vulprotocl.data import default_reveal_dir, load_reveal, stratified_split, subsample
from vulprotocl.metrics import FocalLoss, best_threshold, metrics_from_probs, set_seed, sigmoid


@torch.no_grad()
def predict_gnn(model, loader, device):
    model.eval()
    logits, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        _, logit, _ = model(batch)
        logits.extend(logit.cpu().numpy().tolist())
        ys.extend(batch.y.cpu().numpy().tolist())
    return np.array(logits), np.array(ys)


def train_gnn(train, val, device, epochs, batch_size, seed):
    set_seed(seed)
    tr_ds = CPGVulnDataset(train, use_slice=True, use_domain_graph_feat=True, cache_graphs=True)
    va_ds = CPGVulnDataset(val, use_slice=True, use_domain_graph_feat=True, cache_graphs=True)
    tr_y = np.array(train["vul"])
    num_pos = max(int(tr_y.sum()), 1)
    num_neg = max(len(tr_y) - num_pos, 1)
    weights = np.where(tr_y == 1, float(num_neg) / num_pos, 1.0)
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), True)
    tr_loader = GraphDataLoader(tr_ds, batch_size=batch_size, sampler=sampler)
    va_loader = GraphDataLoader(va_ds, batch_size=batch_size, shuffle=False)

    model = VulCPGPlus(
        NUM_TYPE_BUCKETS,
        NUM_TOKEN_BUCKETS,
        max(EDGE_KIND_TO_ID.values()) + 1,
        use_domain=True,
        node_flag_dim=NODE_FLAG_DIM,
        hidden_dim=192,
        num_layers=4,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    crit = FocalLoss(torch.tensor([num_neg / num_pos], device=device), gamma=2.0)
    best_state, best_f1 = None, -1.0
    for ep in range(epochs):
        model.train()
        for batch in tr_loader:
            batch = batch.to(device)
            _, logits, _ = model(batch)
            loss = crit(logits, batch.y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        logits, ys = predict_gnn(model, va_loader, device)
        _, f1 = best_threshold(ys, sigmoid(logits))
        print(f"[fusion/gnn] epoch {ep+1}/{epochs} val_f1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--max-val", type=int, default=800)
    ap.add_argument("--max-test", type=int, default=1200)
    ap.add_argument("--epochs-seq", type=int, default=8)
    ap.add_argument("--epochs-gnn", type=int, default=6)
    ap.add_argument("--batch-seq", type=int, default=64)
    ap.add_argument("--batch-gnn", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=400)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    full = load_reveal(default_reveal_dir())
    train, val, test = stratified_split(full, args.seed)
    train = subsample(train, args.max_train, args.seed)
    val = subsample(val, args.max_val, args.seed + 1)
    test = subsample(test, args.max_test, args.seed + 2)

    bilstm, vocab = train_bilstm(train, val, device, args.epochs_seq, args.batch_seq, args.max_len, args.seed)
    gnn = train_gnn(train, val, device, args.epochs_gnn, args.batch_gnn, args.seed)

    def seq_loader(split, shuffle=False):
        ds = SeqDS(list(split["func_before"]), np.array(split["vul"]), vocab, args.max_len)
        return DataLoader(ds, batch_size=args.batch_seq, shuffle=shuffle)

    def gnn_loader(split):
        ds = CPGVulnDataset(split, use_slice=True, use_domain_graph_feat=True, cache_graphs=True)
        return GraphDataLoader(ds, batch_size=args.batch_gnn, shuffle=False)

    va_seq_l, va_g_l = predict_seq(bilstm, seq_loader(val), device), predict_gnn(gnn, gnn_loader(val), device)
    te_seq_l, te_g_l = predict_seq(bilstm, seq_loader(test), device), predict_gnn(gnn, gnn_loader(test), device)

    va_ps, va_y = sigmoid(va_seq_l[0]), va_seq_l[1]
    va_pg = sigmoid(va_g_l[0])
    te_ps, te_y = sigmoid(te_seq_l[0]), te_seq_l[1]
    te_pg = sigmoid(te_g_l[0])

    best = {"f1": -1.0, "alpha": 1.0, "th": 0.5}
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = alpha * va_ps + (1 - alpha) * va_pg
        th, f1 = best_threshold(va_y, blended)
        if f1 > best["f1"]:
            best = {"f1": f1, "alpha": float(alpha), "th": float(th)}
    print(f"Best fusion alpha={best['alpha']:.2f} val_f1={best['f1']:.4f} th={best['th']:.2f}", flush=True)

    te_blend = best["alpha"] * te_ps + (1 - best["alpha"]) * te_pg
    m_fuse = metrics_from_probs(te_y, te_blend, best["th"])
    m_seq = metrics_from_probs(te_y, te_ps, best_threshold(va_y, va_ps)[0])
    m_gnn = metrics_from_probs(te_y, te_pg, best_threshold(va_y, va_pg)[0])

    rows = [
        {"config": "bilstm_alone", **m_seq},
        {"config": "spg_domain_alone", **m_gnn},
        {"config": "late_fusion", "alpha": best["alpha"], "threshold": best["th"], **m_fuse},
    ]
    for r in rows:
        print(r, flush=True)

    out = ROOT / "results" / "baselines"
    out.mkdir(exist_ok=True)
    path = out / "results_reveal_fusion.csv"
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved {path}", flush=True)

    torch.save(
        {
            "bilstm": bilstm.state_dict(),
            "gnn": gnn.state_dict(),
            "vocab": vocab,
            "alpha": best["alpha"],
            "threshold": best["th"],
        },
        out / "late_fusion_best.pt",
    )


if __name__ == "__main__":
    main()
