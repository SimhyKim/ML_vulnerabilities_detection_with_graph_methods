
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from vulprotocl.data import set_seed

__all__ = [
    "FocalLoss",
    "set_seed",
    "pr_operating_points",
    "metrics_from_probs",
    "best_threshold",
    "sigmoid",
]


class FocalLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, p, 1 - p).clamp(min=1e-6, max=1.0)
        return (((1 - pt) ** self.gamma) * bce).mean()


def sigmoid(x) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def pr_operating_points(
    y: np.ndarray, probs: np.ndarray, recall_targets=(0.2, 0.4, 0.6)
) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    probs = np.asarray(probs, dtype=float)
    out: Dict[str, float] = {}
    if y.sum() == 0 or len(np.unique(y)) < 2:
        for t in recall_targets:
            out[f"prec_at_rec_{t}"] = float("nan")
        return out
    prec, rec, _ = precision_recall_curve(y, probs)
    for t in recall_targets:
        mask = rec >= float(t)
        out[f"prec_at_rec_{t}"] = float(prec[mask].max()) if mask.any() else 0.0
    return out


def metrics_from_probs(y: np.ndarray, probs: np.ndarray, th: float) -> Dict[str, float]:
    preds = (probs >= th).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(y, preds)
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    tp = int(((preds == 1) & (y == 1)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    try:
        roc = roc_auc_score(y, probs)
    except ValueError:
        roc = float("nan")
    try:
        pra = average_precision_score(y, probs)
    except ValueError:
        pra = float("nan")
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pra),
        "balanced_acc": float(0.5 * (tpr + tnr)),
        **pr_operating_points(y, probs),
    }


def best_threshold(y: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    best_f1, best_th = -1.0, 0.5
    for th in np.linspace(0.05, 0.95, 37):
        f1 = precision_recall_fscore_support(
            y, (probs >= th).astype(int), average="binary", zero_division=0
        )[2]
        if f1 > best_f1:
            best_f1, best_th = float(f1), float(th)
    return best_th, best_f1
