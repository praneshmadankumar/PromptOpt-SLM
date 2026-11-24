"""Latency, throughput, token and memory measurement (Module 6 helpers)."""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LatencyStats:
    n_calls: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    throughput_per_s: float

    def to_dict(self) -> dict:
        return asdict(self)


class LatencyTracker:
    """Records wall-clock durations of repeated calls.

    Usage:  with tracker.measure(): run_something()
    """

    def __init__(self) -> None:
        self._durations_ms: List[float] = []

    class _Span:
        def __init__(self, tracker: "LatencyTracker") -> None:
            self._t = tracker

        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            self._t._durations_ms.append((time.perf_counter() - self._start) * 1000.0)
            return False  # never swallow exceptions

    def measure(self) -> "LatencyTracker._Span":
        return LatencyTracker._Span(self)

    def stats(self) -> Optional[LatencyStats]:
        if not self._durations_ms:
            return None
        d = np.asarray(self._durations_ms)
        total_s = d.sum() / 1000.0
        return LatencyStats(n_calls=len(d), mean_ms=float(d.mean()),
                            p50_ms=float(np.percentile(d, 50)), p95_ms=float(np.percentile(d, 95)),
                            throughput_per_s=float(len(d) / total_s) if total_s > 0 else float("inf"))


def memory_snapshot() -> Dict[str, Optional[float]]:
    """Python heap (tracemalloc, MB) and GPU memory when torch+CUDA exist."""
    snap: Dict[str, Optional[float]] = {"python_heap_mb": None, "python_peak_mb": None,
                                        "gpu_allocated_mb": None}
    if tracemalloc.is_tracing():
        cur, peak = tracemalloc.get_traced_memory()
        snap["python_heap_mb"] = round(cur / 2**20, 3)
        snap["python_peak_mb"] = round(peak / 2**20, 3)
    try:
        import torch
        if torch.cuda.is_available():
            snap["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
    except ImportError:
        pass
    return snap
