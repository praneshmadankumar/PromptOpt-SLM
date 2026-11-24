"""Prompt representation and candidate generation (Module 2, part 1).

A prompt configuration is pi = (instruction, verbalizer, demonstrations,
calibration). Candidate instructions come from a small grammar; verbalizers
are expanded from synonym lists and filtered for tokenizer validity.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Demo = Tuple[str, int]


@dataclass
class PromptConfig:
    """pi = (I, V, D, c) plus formatting fields."""
    instruction: str
    verbalizer: Dict[int, str]
    demos: List[Demo] = field(default_factory=list)
    calibration: str = "contextual"          # 'none' | 'contextual' | 'prior'
    input_prefix: str = "Text:"
    output_prefix: str = "Label:"
    sep: str = "\n\n"
    template_id: str = "base"

    def __post_init__(self) -> None:
        if self.calibration not in ("none", "contextual", "prior"):
            raise ValueError(f"unknown calibration '{self.calibration}'")
        if len(set(self.verbalizer.values())) != len(self.verbalizer):
            raise ValueError("verbalizer words must be distinct")

    def signature(self) -> str:
        """Stable identifier used for caching, leaderboard and version history."""
        payload = json.dumps({
            "i": self.instruction, "v": sorted(self.verbalizer.items()),
            "d": [list(d) for d in self.demos], "c": self.calibration,
        }, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verbalizer"] = {str(k): v for k, v in self.verbalizer.items()}
        d["demos"] = [list(x) for x in self.demos]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PromptConfig":
        d = dict(d)
        d["verbalizer"] = {int(k): v for k, v in d["verbalizer"].items()}
        d["demos"] = [(t, int(y)) for t, y in d.get("demos", [])]
        return cls(**d)

    def with_demos(self, demos: Sequence[Demo]) -> "PromptConfig":
        d = self.to_dict()
        d["demos"] = [list(x) for x in demos]
        return PromptConfig.from_dict(d)


def build_prompt(cfg: PromptConfig, text: str, demos: Optional[Sequence[Demo]] = None) -> str:
    """Assemble instruction, demonstrations and an open answer slot."""
    demos = cfg.demos if demos is None else demos
    parts = [cfg.instruction.strip()]
    for x, y in demos:
        parts.append(f"{cfg.input_prefix} {x.strip()}\n{cfg.output_prefix} {cfg.verbalizer[y]}")
    parts.append(f"{cfg.input_prefix} {text.strip()}\n{cfg.output_prefix}")
    return cfg.sep.join(parts)


INSTRUCTION_VERBS = ("Classify", "Decide", "Determine")


def generate_instructions(task_noun: str, class_names: Sequence[str]) -> List[str]:
    """Verb x phrasing x label-list grammar (8 to 18 candidates)."""
    labels = ", ".join(class_names)
    out = []
    for v in INSTRUCTION_VERBS:
        out.append(f"{v} {task_noun}.")
        out.append(f"{v} {task_noun}. Answer with one of: {labels}.")
        out.append(f"Read the text and {v.lower()} {task_noun}. The possible labels are: {labels}.")
    return out


def generate_verbalizers(
    class_names: Sequence[str],
    synonyms: Dict[str, Sequence[str]],
    first_token: Callable[[str], object],
    max_candidates: int = 8,
    seed: int = 0,
) -> List[Dict[int, str]]:
    """Expand class names with synonyms and keep combinations whose
    first tokens are pairwise distinct. The default mapping is always first.

    `first_token` maps a label word to the id of its first token for the
    target tokenizer. Multi-token words are allowed; their full sequence is
    scored by the backend, so they are not rejected here.
    """
    rng = random.Random(seed)
    options = [[name] + [w for w in synonyms.get(name, []) if w != name] for name in class_names]
    default = {i: name for i, name in enumerate(class_names)}
    combos = []
    for combo in itertools.product(*options):
        firsts = [first_token(w) for w in combo]
        if len(set(firsts)) == len(firsts):
            combos.append({i: w for i, w in enumerate(combo)})
    rng.shuffle(combos)
    combos = [default] + [c for c in combos if c != default]
    return combos[:max_candidates]
