"""EXPERIMENTAL - Adaptive Prompt Evolution (proposed extension).

A small evolutionary loop over prompt configurations. Mutation and
recombination are rule-based (no LLM proposer), so the method stays
offline and cheap. Children are screened on a 25 percent validation subset,
and only the best few receive a full evaluation.

This is a proposed extension of the original framework. It has NOT been
shown to outperform successive halving on the bundled data; comparisons are
reported as pending evaluation in the report unless measured by run_demo.py.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Sequence

from core.prompt_generator import PromptConfig, Demo
from core.prompt_optimizer import PromptOptimizer

CALIBRATIONS = ("none", "contextual", "prior")


@dataclass
class EvolutionResult:
    best: PromptConfig
    best_J: float
    generations: List[dict] = field(default_factory=list)


class AdaptiveEvolver:
    def __init__(self, instructions: Sequence[str], verbalizers: Sequence[dict],
                 demo_pool: Sequence[Demo], n_demos: int, seed: int = 0,
                 mutation_rate: float = 0.3):
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        self.instructions = list(instructions)
        self.verbalizers = list(verbalizers)
        self.demo_pool = list(demo_pool)
        self.n_demos = n_demos
        self.rng = random.Random(seed)
        self.mutation_rate = mutation_rate

    def mutate(self, cfg: PromptConfig) -> PromptConfig:
        child = cfg.to_dict()
        if self.rng.random() < self.mutation_rate:
            child["instruction"] = self.rng.choice(self.instructions)
            child["template_id"] = "mut-instr"
        if self.rng.random() < self.mutation_rate:
            child["verbalizer"] = {int(k): v for k, v in self.rng.choice(self.verbalizers).items()}
            child["template_id"] = "mut-verb"
        if self.rng.random() < self.mutation_rate and child["demos"]:
            demos = [tuple(d) for d in child["demos"]]
            i = self.rng.randrange(len(demos))
            demos[i] = self.rng.choice(self.demo_pool)
            child["demos"] = [list(d) for d in demos]
            child["template_id"] = "mut-demo"
        if self.rng.random() < self.mutation_rate * 0.5:
            child["calibration"] = self.rng.choice(CALIBRATIONS)
            child["template_id"] = "mut-calib"
        return PromptConfig.from_dict(child)

    def recombine(self, a: PromptConfig, b: PromptConfig) -> PromptConfig:
        """Instruction and verbalizer from one parent, demonstrations mixed from both."""
        pool = list({tuple(d) for d in a.demos} | {tuple(d) for d in b.demos})
        self.rng.shuffle(pool)
        demos = pool[: self.n_demos] if len(pool) >= self.n_demos else pool
        parent_a, parent_b = (a, b) if self.rng.random() < 0.5 else (b, a)
        child = PromptConfig(instruction=parent_a.instruction, verbalizer=dict(parent_a.verbalizer),
                             demos=[(t, y) for t, y in demos], calibration=parent_b.calibration,
                             template_id="recomb")
        return child

    def evolve(self, optimizer: PromptOptimizer, parents: Sequence[PromptConfig],
               generations: int = 3, offspring: int = 8, top_k: int = 3) -> EvolutionResult:
        population = list(parents)
        best_cfg, best_J = None, float("-inf")
        log = []
        for g in range(generations):
            children = []
            for _ in range(offspring):
                a, b = self.rng.sample(population, 2) if len(population) > 1 else (population[0], population[0])
                children.append(self.recombine(a, b) if self.rng.random() < 0.5 else self.mutate(a))
            screened = sorted(((optimizer.stage_score(c, 0.25), c) for c in children),
                              key=lambda t: t[0], reverse=True)[:top_k]
            evals = [(optimizer.full_evaluate(c, stage=f"gen{g}"), c) for _, c in screened]
            evals.sort(key=lambda t: t[0].J, reverse=True)
            if evals[0][0].J > best_J:
                best_J, best_cfg = evals[0][0].J, evals[0][1]
            population = [c for _, c in evals] + population[:top_k]
            log.append({"generation": g, "best_J": round(evals[0][0].J, 4),
                        "best_f1": round(evals[0][0].f1, 4), "children_screened": len(children)})
        return EvolutionResult(best=best_cfg, best_J=best_J, generations=log)
