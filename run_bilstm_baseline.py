
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vulprotocl.data import default_reveal_dir, load_reveal, stratified_split, subsample
from vulprotocl.metrics import FocalLoss, best_threshold, metrics_from_probs, set_seed, sigmoid
from vulprotocl.seq import BiLSTM, SeqDS, build_vocab, predict_seq

__all__ = ["BiLSTM", "SeqDS", "build_vocab"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=400)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = load_reveal(default_reveal_dir())
    train, val, test = stratified_split(full, args.seed)
    train = subsample(train, args.max_train, args.seed)
    val = subsample(val, args.max_val, args.seed + 1)
    test = subsample(test, args.max_test, args.seed + 2)

    tr_texts = list(train["func_before"])
    vocab = build_vocab(tr_texts)
    tr_y = np.array(train["vul"])
    va_y = np.array(val["vul"])
    te_y = np.array(test["vul"])
    tr_ds = SeqDS(tr_texts, tr_y, vocab, args.max_len)
    va_ds = SeqDS(list(val["func_before"]), va_y, vocab, args.max_len)
    te_ds = SeqDS(list(test["func_before"]), te_y, vocab, args.max_len)
    num_pos = max(int(tr_y.sum()), 1)
    num_neg = max(len(tr_y) - num_pos, 1)
    weights = np.where(tr_y == 1, float(num_neg) / num_pos, 1.0)
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), True)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size)

    model = BiLSTM(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    crit = FocalLoss(torch.tensor([num_neg / num_pos], device=device), gamma=2.0)
    best_state, best_f1, best_th = None, -1.0, 0.5
    for ep in range(args.epochs):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        logits, labels = predict_seq(model, va_loader, device)
        th, f1 = best_threshold(labels, sigmoid(logits))
        print(f"[bilstm] epoch {ep+1}/{args.epochs} val_f1={f1:.4f} th={th:.2f}", flush=True)
        if f1 > best_f1:
            best_f1, best_th = f1, th
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    te_logits, _ = predict_seq(model, te_loader, device)
    m = metrics_from_probs(te_y, sigmoid(te_logits), best_th)
    print({"config": "bilstm", **m}, flush=True)
    out = ROOT / "results" / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"results_reveal_bilstm_seed{args.seed}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config", *m.keys()])
        w.writeheader()
        w.writerow({"config": "bilstm", **m})
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
