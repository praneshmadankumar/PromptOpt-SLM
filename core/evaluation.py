"""Metrics for classification quality, calibration and uncertainty.

Implemented: accuracy, macro/weighted precision-recall-F1, confusion matrix,
expected calibration error (ECE, equal-width bins), multi-class Brier score,
and a percentile bootstrap confidence interval.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                           labels: Optional[Sequence[int]] = None) -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same length")
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                       labels=labels, zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted",
                                                       labels=labels, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(p_m), "recall_macro": float(r_m), "f1_macro": float(f_m),
            "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f_w)}


def confusion(y_true: Sequence[int], y_pred: Sequence[int], K: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(K)))


def expected_calibration_error(probs: np.ndarray, y_true: Sequence[int],
                               n_bins: int = 10) -> Tuple[float, pd.DataFrame]:
    """ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)| over equal-width bins."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        n_b = int(mask.sum())
        if n_b:
            acc_b, conf_b = float(correct[mask].mean()), float(conf[mask].mean())
            ece += (n_b / len(conf)) * abs(acc_b - conf_b)
        else:
            acc_b, conf_b = float("nan"), float("nan")
        rows.append({"bin": b, "lower": lo, "upper": hi, "count": n_b,
                     "accuracy": acc_b, "confidence": conf_b})
    return float(ece), pd.DataFrame(rows)


def brier_score(probs: np.ndarray, y_true: Sequence[int]) -> float:
    """Multi-class Brier score: mean over examples of sum_k (p_k - 1[y = k])^2."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(probs.shape[1])[np.asarray(y_true)]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def bootstrap_ci(y_true: Sequence[int], y_pred: Sequence[int],
                 metric: Callable[[Sequence[int], Sequence[int]], float] = accuracy_score,
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """Point estimate and percentile CI from example-level bootstrap resampling."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y_true, y_pred)), float(lo), float(hi)
