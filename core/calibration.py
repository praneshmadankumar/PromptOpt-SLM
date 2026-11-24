"""Output-side calibration (Module 2 / Module 3).

Contextual calibration (Zhao et al., 2021): estimate the model's prior
label bias from content-free inputs and divide it out. Prior matching is a
cheaper alternative that equalises the average predicted distribution.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

CONTENT_FREE = ("N/A", "", "[MASK]")


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = np.atleast_2d(np.asarray(logits, dtype=float))
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def estimate_bias(score_fn: Callable[[Sequence[str]], np.ndarray],
                  content_free: Sequence[str] = CONTENT_FREE) -> np.ndarray:
    """Average predicted class distribution on content-free inputs.

    `score_fn(texts)` must return an (n, K) matrix of unnormalised class
    scores for the given texts under the *same* prompt configuration.
    """
    probs = softmax_rows(score_fn(list(content_free)))
    bias = probs.mean(axis=0)
    return np.clip(bias, 1e-8, None) / np.clip(bias, 1e-8, None).sum()


def contextual_calibrate(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """p~(y|x) proportional to p(y|x) / p_bias(y), renormalised."""
    p = np.asarray(probs, dtype=float) / np.asarray(bias, dtype=float)
    return p / p.sum(axis=1, keepdims=True)


def prior_match(probs: np.ndarray, iters: int = 50) -> np.ndarray:
    """Rescale class columns so the mean prediction is uniform."""
    p = np.asarray(probs, dtype=float)
    w = np.ones(p.shape[1])
    for _ in range(iters):
        q = p * w
        q /= q.sum(axis=1, keepdims=True)
        w *= (1.0 / p.shape[1]) / q.mean(axis=0)
    q = p * w
    return q / q.sum(axis=1, keepdims=True)
