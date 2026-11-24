"""Module 2 - Adaptive prompt optimisation (search controller).

Objective (interpretable, all terms on the validation split only):

    J(pi) = MacroF1(pi) - lambda * Std_sigma[MacroF1(pi_sigma)]
            - mu * tokens(pi) / token_ref - nu * latency_ms(pi) / latency_ref

The first two terms follow the original report (Section 3.3). The last two
are proposed extensions; they are set to small defaults and can be zeroed.

Search: random candidate pool + successive halving (25/50/100 percent of the
validation set), finishing with a stability check over demonstration
permutations. A budget caps the number of forward passes; exceeding it raises
BudgetExceededError so that a run never silently overruns.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score

from core.model_engine import ScoringBackend, SLMClassifier
from core.prompt_generator import PromptConfig, build_prompt
from core.resource_monitor import LatencyTracker


class BudgetExceededError(RuntimeError):
    """Raised when the forward-pass budget would be exceeded."""


@dataclass
class Budget:
    max_forward_passes: int
    used: int = 0

    def charge(self, n: int) -> None:
        if self.used + n > self.max_forward_passes:
            raise BudgetExceededError(
                f"budget {self.max_forward_passes} forward passes exhausted (used {self.used}, need {n})")
        self.used += n

    @property
    def remaining(self) -> int:
        return self.max_forward_passes - self.used


@dataclass(frozen=True)
class ObjectiveWeights:
    stability_lambda: float = 0.5
    token_mu: float = 0.02
    token_ref: float = 200.0
    latency_nu: float = 0.0          # latency reported but not weighted by default
    latency_ref_ms: float = 100.0


def objective(f1: float, std: float, tokens: float, latency_ms: float,
              w: ObjectiveWeights) -> float:
    return (f1 - w.stability_lambda * std
            - w.token_mu * tokens / w.token_ref
            - w.latency_nu * latency_ms / w.latency_ref_ms)


@dataclass
class Evaluation:
    signature: str
    f1: float
    std: float
    tokens: float
    latency_ms: float
    J: float
    forward_passes: int
    stage: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OptimizationResult:
    best: PromptConfig
    best_eval: Evaluation
    leaderboard: List[dict]
    history: List[dict]
    forward_passes: int
    wall_seconds: float


class PromptOptimizer:
    """Scores prompt configurations on a validation split under a budget."""

    def __init__(self, backend: ScoringBackend, val_texts: Sequence[str], val_labels: Sequence[int],
                 budget: Budget, weights: ObjectiveWeights = ObjectiveWeights(),
                 n_permutations: int = 4, seed: int = 0,
                 on_version: Optional[Callable[[dict], None]] = None):
        if len(val_texts) != len(val_labels) or not val_texts:
            raise ValueError("validation texts and labels must be non-empty and aligned")
        self.backend = backend
        self.budget = budget
        self.weights = weights
        self.n_perm = n_permutations
        self.seed = seed
        self.rng = random.Random(seed)
        order = list(range(len(val_texts)))
        self.rng.shuffle(order)                     # fixed shuffled order -> nested subsets
        self.val_texts = [val_texts[i] for i in order]
        self.val_labels = [int(val_labels[i]) for i in order]
        self.K = len(set(self.val_labels))
        self.history: List[dict] = []
        self._on_version = on_version

    # ------------------------------------------------------------ scoring
    def _macro_f1(self, cfg: PromptConfig, n: int) -> Tuple[float, float]:
        """Macro-F1 on the first n validation items; returns (f1, latency_ms)."""
        self.budget.charge(n)
        clf = SLMClassifier(self.backend, cfg)
        texts, labels = self.val_texts[:n], self.val_labels[:n]
        tracker = LatencyTracker()
        with tracker.measure():
            preds = clf.predict(texts)
        f1 = f1_score(labels, preds, average="macro", labels=list(range(len(cfg.verbalizer))),
                      zero_division=0)
        lat = tracker.stats().mean_ms
        return float(f1), float(lat)

    def _tokens(self, cfg: PromptConfig) -> float:
        sample = self.val_texts[:min(8, len(self.val_texts))]
        counts = [self.backend.count_tokens(build_prompt(cfg, t)) for t in sample]
        return float(np.mean(counts))

    def stage_score(self, cfg: PromptConfig, frac: float) -> float:
        n = max(2 * self.K, int(round(frac * len(self.val_texts))))
        f1, _ = self._macro_f1(cfg, n)
        return f1

    def full_evaluate(self, cfg: PromptConfig, stage: str = "final") -> Evaluation:
        """Full validation macro-F1 plus stability over demonstration orders."""
        n = len(self.val_texts)
        f1, lat = self._macro_f1(cfg, n)
        perm_scores = []
        rng = random.Random(self.seed)
        for _ in range(self.n_perm):
            demos = list(cfg.demos)
            rng.shuffle(demos)
            perm_cfg = cfg.with_demos(demos)
            perm_scores.append(self._macro_f1(perm_cfg, n)[0])
        std = float(np.std(perm_scores))
        tokens = self._tokens(cfg)
        J = objective(f1, std, tokens, lat, self.weights)
        ev = Evaluation(signature=cfg.signature(), f1=f1, std=std, tokens=tokens,
                        latency_ms=lat, J=J, forward_passes=n * (1 + self.n_perm), stage=stage)
        self._record_version(cfg, ev)
        return ev

    def _record_version(self, cfg: PromptConfig, ev: Evaluation) -> None:
        entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "signature": cfg.signature(), "J": round(ev.J, 4), "f1": round(ev.f1, 4),
                 "std": round(ev.std, 4), "template_id": cfg.template_id,
                 "calibration": cfg.calibration, "n_demos": len(cfg.demos)}
        self.history.append(entry)
        if self._on_version:
            self._on_version({**entry, "config": cfg.to_dict()})

    # ------------------------------------------------------------ search
    def successive_halving(self, candidates: Sequence[PromptConfig],
                           fractions: Sequence[float] = (0.25, 0.5, 1.0),
                           shortlist: int = 3) -> OptimizationResult:
        if fractions[-1] != 1.0:
            raise ValueError("final fraction must be 1.0")
        t0 = time.perf_counter()
        start_used = self.budget.used
        pool = list(candidates)
        last: Dict[str, Tuple[float, str]] = {}
        stages = []
        for s, frac in enumerate(fractions[:-1]):
            scored = sorted(((self.stage_score(c, frac), c) for c in pool),
                            key=lambda t: t[0], reverse=True)
            for score, c in scored:
                last[c.signature()] = (score, f"stage{s}@{frac:.2f}")
            stages.append({"stage": s, "fraction": frac, "n_candidates": len(pool),
                           "best": round(scored[0][0], 4),
                           "median": round(float(np.median([x for x, _ in scored])), 4)})
            pool = [c for _, c in scored[:max(shortlist, len(scored) // 2)]]
        # final full-validation stage on the shortlist
        finalists = []
        for c in pool[:shortlist]:
            ev = self.full_evaluate(c, stage="final")
            finalists.append((ev, c))
            last[c.signature()] = (ev.f1, "final")
        finalists.sort(key=lambda t: t[0].J, reverse=True)
        best_ev, best = finalists[0]
        board = self._leaderboard(candidates, last, {ev.signature: ev for ev, _ in finalists})
        return OptimizationResult(best=best, best_eval=best_ev, leaderboard=board,
                                  history=stages + [{"stage": "final", "J": round(best_ev.J, 4)}],
                                  forward_passes=self.budget.used - start_used,
                                  wall_seconds=time.perf_counter() - t0)

    def random_search_full(self, candidates: Sequence[PromptConfig]) -> OptimizationResult:
        """Baseline: every candidate on the full validation set (no halving)."""
        t0 = time.perf_counter()
        start_used = self.budget.used
        evals = [(self.full_evaluate(c, stage="full"), c) for c in candidates]
        evals.sort(key=lambda t: t[0].J, reverse=True)
        best_ev, best = evals[0]
        last = {ev.signature: (ev.f1, "full") for ev, _ in evals}
        board = self._leaderboard(candidates, last, {ev.signature: ev for ev, _ in evals})
        return OptimizationResult(best=best, best_eval=best_ev, leaderboard=board,
                                  history=[{"stage": "full", "n_candidates": len(candidates)}],
                                  forward_passes=self.budget.used - start_used,
                                  wall_seconds=time.perf_counter() - t0)

    @staticmethod
    def _leaderboard(candidates, last, finals) -> List[dict]:
        rows = []
        for c in candidates:
            sig = c.signature()
            ev = finals.get(sig)
            score, stage = last.get(sig, (float("nan"), "not reached"))
            rows.append({"signature": sig, "template_id": c.template_id,
                         "calibration": c.calibration, "stage": stage,
                         "score": round(score, 4),
                         "J": round(ev.J, 4) if ev else None,
                         "f1": round(ev.f1, 4) if ev else None,
                         "std": round(ev.std, 4) if ev else None})
        rows.sort(key=lambda r: (r["J"] is None, -(r["J"] or 0), -(r["score"] or 0)))
        for rank, r in enumerate(rows, 1):
            r["rank"] = rank
        return rows
