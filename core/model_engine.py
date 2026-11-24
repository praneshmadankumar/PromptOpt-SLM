"""Module 3 - Multi-SLM classification engine (scoring backends).

Two backends share one abstract interface (polymorphism):

* SimulatedBackend - DETERMINISTIC SIMULATION. It is a lexicon heuristic with
  injected label biases, written so that the pipeline, calibration and
  search logic can run offline. It is NOT a language model and its numbers
  must never be reported as SLM results.
* HFBackend - real causal-LM scoring with Hugging Face Transformers. It
  scores each label word as a full token sequence (teacher forcing), so
  multi-token verbalizers are handled correctly. It requires torch and
  transformers and is untested in the environment where this was built.
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.calibration import contextual_calibrate, estimate_bias, prior_match, softmax_rows
from core.prompt_generator import PromptConfig, Demo, build_prompt


class ModelUnavailableError(RuntimeError):
    """Raised when a real model backend cannot be created."""


class ScoringBackend(ABC):
    """Maps prompts to unnormalised class scores (n_prompts x K)."""

    name: str = "abstract"
    is_simulated: bool = False

    @abstractmethod
    def label_logits(self, prompts: Sequence[str], verbalizer: Dict[int, str]) -> np.ndarray:
        ...

    def count_tokens(self, text: str) -> int:
        """Approximate token count (whitespace units for the simulator)."""
        return len(text.split())

    def describe(self) -> dict:
        return {"name": self.name, "simulated": self.is_simulated}


_TOKEN = re.compile(r"[a-z0-9']+")


def build_lexicon(texts: Sequence[str], labels: Sequence[int], K: int,
                  top_k: int = 8, min_count: int = 3) -> Dict[int, set]:
    """Words most associated with each class in a *support* set (no test data)."""
    totals: Counter = Counter()
    per_class = [Counter() for _ in range(K)]
    for t, y in zip(texts, labels):
        toks = set(_TOKEN.findall(t.lower()))
        totals.update(toks)
        per_class[y].update(toks)
    lex: Dict[int, set] = {k: set() for k in range(K)}
    for k in range(K):
        scored = []
        for w, c in per_class[k].items():
            if c < min_count:
                continue
            purity = c / totals[w]
            scored.append((purity * c, w))
        scored.sort(reverse=True)
        lex[k] = {w for _, w in scored[:top_k]}
    return lex


class SimulatedBackend(ScoringBackend):
    """SIMULATED scorer - not a language model.

    logit_k = evidence_weight * lexicon_hits_k(query)
            + demo_weight * recency-weighted count of label word k in demos
            + prior_k                       (fixed label prior, emulates bias)
            + noise_k(prompt)               (deterministic per prompt)
    The demo and prior terms reproduce the label-bias effects described in
    Zhao et al. (2021) so that calibration has something to correct.
    """

    name = "simulated-lexicon"
    is_simulated = True

    def __init__(self, lexicon: Dict[int, set], prior: Optional[np.ndarray] = None,
                 evidence_weight: float = 1.2, demo_weight: float = 1.0,
                 noise_scale: float = 0.15, recency_decay: float = 0.8,
                 input_marker: str = "Text:", output_marker: str = "Label:"):
        self.lexicon = lexicon
        self.K = len(lexicon)
        self.prior = np.zeros(self.K) if prior is None else np.asarray(prior, dtype=float)
        self.evidence_weight = evidence_weight
        self.demo_weight = demo_weight
        self.noise_scale = noise_scale
        self.recency_decay = recency_decay
        self.input_marker = input_marker
        self.output_marker = output_marker

    def _query(self, prompt: str) -> str:
        return prompt.rsplit(self.input_marker, 1)[-1].rsplit(self.output_marker, 1)[0]

    def _scores_one(self, prompt: str, verbalizer: Dict[int, str]) -> np.ndarray:
        query_tokens = set(_TOKEN.findall(self._query(prompt).lower()))
        evidence = np.array([len(query_tokens & self.lexicon[k]) for k in range(self.K)], dtype=float)
        # demonstrations: label words appear after 'Label:'; recent demos weigh more
        demo_bias = np.zeros(self.K)
        occurrences = [(m.start(), m.group(1)) for m in re.finditer(r"Label: (\S+)", prompt)]
        n = len(occurrences)
        for rank, (_, word) in enumerate(occurrences):
            weight = self.recency_decay ** (n - 1 - rank)
            for k, v in verbalizer.items():
                if word == v:
                    demo_bias[k] += weight
        digest = hashlib.md5(prompt.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        noise = rng.normal(0.0, self.noise_scale, size=self.K)
        return (self.evidence_weight * evidence + self.demo_weight * demo_bias
                + self.prior + noise)

    def label_logits(self, prompts: Sequence[str], verbalizer: Dict[int, str]) -> np.ndarray:
        return np.stack([self._scores_one(p, verbalizer) for p in prompts])


class HFBackend(ScoringBackend):
    """Real causal-LM scorer (Hugging Face Transformers, PyTorch).

    Each label word is scored as the sum of log-probabilities of its full
    token sequence given the prompt, so multi-token labels are supported.
    Prompts longer than `max_length` tokens keep their *tail* so that the
    query and answer slot are never truncated away.
    Cost: one forward pass per (prompt, label). KV-cache reuse is future work.
    """

    name = "hf-causal-lm"
    is_simulated = False

    def __init__(self, model_name: str, device: Optional[str] = None,
                 max_length: int = 2048, load_in_4bit: bool = False):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ModelUnavailableError("torch/transformers not installed; use SimulatedBackend "
                                        "or install requirements.txt") from exc
        self._torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tok = AutoTokenizer.from_pretrained(model_name)
            kwargs = {"torch_dtype": torch.float16 if self.device == "cuda" else torch.float32}
            if load_in_4bit and self.device == "cuda":
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except Exception as exc:  # network, missing checkpoint, OOM, ...
            raise ModelUnavailableError(f"could not load '{model_name}': {exc}") from exc
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model.to(self.device).eval()
        self.name = f"hf:{model_name}"

    def count_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def label_logits(self, prompts: Sequence[str], verbalizer: Dict[int, str]) -> np.ndarray:
        torch = self._torch
        K = len(verbalizer)
        out = np.zeros((len(prompts), K))
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                ctx = self.tok(prompt, add_special_tokens=False).input_ids
                ctx = ctx[-self.max_length:]                       # keep the tail
                for k in sorted(verbalizer):
                    lab = self.tok(" " + verbalizer[k], add_special_tokens=False).input_ids
                    ids = torch.tensor([ctx + lab], device=self.device)
                    logits = self.model(ids).logits[0].float()
                    logp = torch.log_softmax(logits, dim=-1)
                    n = len(ctx)
                    tgt = ids[0, n:]
                    out[i, k] = float(logp[n - 1:-1].gather(1, tgt[:, None]).sum())
        return out


class SLMClassifier:
    """Prompt-configured classifier around any ScoringBackend."""

    def __init__(self, backend: ScoringBackend, cfg: PromptConfig):
        self.backend = backend
        self.cfg = cfg
        self._bias: Optional[np.ndarray] = None
        self.forward_passes = 0

    @property
    def K(self) -> int:
        return len(self.cfg.verbalizer)

    def _logits(self, prompts: List[str]) -> np.ndarray:
        self.forward_passes += len(prompts)
        return self.backend.label_logits(prompts, self.cfg.verbalizer)

    def bias(self) -> np.ndarray:
        """Contextual-calibration bias for this configuration (cached)."""
        if self._bias is None:
            self._bias = estimate_bias(
                lambda cf: self._logits([build_prompt(self.cfg, t) for t in cf]))
        return self._bias

    def predict_proba(self, texts: Sequence[str],
                      demos_per_text: Optional[Sequence[Sequence[Demo]]] = None) -> np.ndarray:
        if demos_per_text is None:
            prompts = [build_prompt(self.cfg, t) for t in texts]
        else:
            prompts = [build_prompt(self.cfg, t, d) for t, d in zip(texts, demos_per_text)]
        probs = softmax_rows(self._logits(prompts))
        if self.cfg.calibration == "contextual":
            probs = contextual_calibrate(probs, self.bias())
        elif self.cfg.calibration == "prior":
            probs = prior_match(probs)
        return probs

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=1)
