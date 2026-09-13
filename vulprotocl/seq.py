
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from vulprotocl.metrics import FocalLoss, best_threshold, set_seed, sigmoid


def tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^\w]+", (text or "").lower()) if t]


def build_vocab(texts: List[str], min_freq: int = 2) -> Dict[str, int]:
    c: Counter = Counter()
    for t in texts:
        c.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, n in c.items():
        if n >= min_freq:
            vocab[w] = len(vocab)
    return vocab


class SeqDS(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = np.asarray(labels).astype(np.float32)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        ids = [self.vocab.get(w, 1) for w in tokenize(self.texts[i])][: self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[i])


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb=100, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(
            emb, hidden, num_layers=layers, bidirectional=True, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        mask = (x != 0).float().unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.fc(pooled).squeeze(-1)


@torch.no_grad()
def predict_seq(model, loader, device):
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.extend(model(x.to(device)).cpu().numpy().tolist())
        ys.extend(y.numpy().tolist())
    return np.array(logits), np.array(ys)


def train_bilstm(train, val, device, epochs, batch_size, max_len, seed):
    set_seed(seed)
    tr_texts = list(train["func_before"])
    vocab = build_vocab(tr_texts)
    tr_y = np.array(train["vul"])
    va_y = np.array(val["vul"])
    tr_ds = SeqDS(tr_texts, tr_y, vocab, max_len)
    va_ds = SeqDS(list(val["func_before"]), va_y, vocab, max_len)
    num_pos = max(int(tr_y.sum()), 1)
    num_neg = max(len(tr_y) - num_pos, 1)
    weights = np.where(tr_y == 1, float(num_neg) / num_pos, 1.0)
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), True)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler)
    va_loader = DataLoader(va_ds, batch_size=batch_size)

    model = BiLSTM(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    crit = FocalLoss(torch.tensor([num_neg / num_pos], device=device), gamma=2.0)
    best_state, best_f1 = None, -1.0
    for ep in range(epochs):
        model.train()
        print(f"[bilstm] epoch {ep+1}/{epochs} {len(tr_loader)} batches", flush=True)
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        logits, ys = predict_seq(model, va_loader, device)
        _, f1 = best_threshold(ys, sigmoid(logits))
        print(f"[bilstm] epoch {ep+1}/{epochs} val_f1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, vocab
