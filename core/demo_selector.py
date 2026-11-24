"""Demonstration selection (Module 2, part 2).

Strategies: random, class-balanced random, and class-balanced MMR
(maximal marginal relevance). Embeddings are pluggable: the default is a
TF-IDF embedder (offline); a sentence-transformers embedder is provided and
loaded lazily only when explicitly requested.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from core.prompt_generator import Demo


class Embedder(Protocol):
    def fit(self, texts: Sequence[str]) -> None: ...
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...   # L2-normalised rows


class TfidfEmbedder:
    """Offline, dependency-light embedder. Cosine similarity = dot product."""

    def __init__(self) -> None:
        self._vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))

    def fit(self, texts: Sequence[str]) -> None:
        self._vec.fit(list(texts))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._vec.transform(list(texts)).toarray()


class SentenceTransformerEmbedder:
    """Optional higher-quality embedder (requires sentence-transformers)."""

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # lazy import
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed; use TfidfEmbedder") from exc
        self._model = SentenceTransformer(name)

    def fit(self, texts: Sequence[str]) -> None:  # pretrained: nothing to fit
        return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True))


class DemoSelector:
    """Selects demonstrations from a fixed support set."""

    def __init__(self, support: Sequence[Demo], n_classes: int, embedder: Optional[Embedder] = None):
        if not support:
            raise ValueError("support set is empty")
        self.support = list(support)
        self.K = n_classes
        self.embedder = embedder or TfidfEmbedder()
        texts = [x for x, _ in self.support]
        self.embedder.fit(texts)
        self.emb = self.embedder.encode(texts)

    # baselines ---------------------------------------------------------
    def random(self, m: int, rng: random.Random) -> List[Demo]:
        return rng.sample(self.support, min(m, len(self.support)))

    def balanced(self, m: int, rng: random.Random) -> List[Demo]:
        per_class = max(1, m // self.K)
        chosen: List[Demo] = []
        for c in range(self.K):
            pool = [d for d in self.support if d[1] == c]
            chosen += rng.sample(pool, min(per_class, len(pool)))
        rng.shuffle(chosen)
        return chosen

    # proposed: class-balanced MMR --------------------------------------
    def mmr(self, query: str, m: int, alpha: float = 0.7) -> List[Demo]:
        """Greedy MMR with a per-class cap of ceil(m / K).

        score(d) = alpha * cos(e(d), e(x)) - (1 - alpha) * max_{d' in S} cos(e(d), e(d'))
        Selected demonstrations are returned most-similar-last.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        q = self.embedder.encode([query])[0]
        rel = self.emb @ q
        cap = math.ceil(m / self.K)
        target = min(m, len(self.support))
        chosen: List[int] = []
        counts = [0] * self.K
        while len(chosen) < target:
            best, best_score = None, -np.inf
            for i, (_, y) in enumerate(self.support):
                if i in chosen or counts[y] >= cap:
                    continue
                redundancy = max((float(self.emb[i] @ self.emb[j]) for j in chosen), default=0.0)
                score = alpha * float(rel[i]) - (1 - alpha) * redundancy
                if score > best_score:
                    best, best_score = i, score
            if best is None:
                break
            chosen.append(best)
            counts[self.support[best][1]] += 1
        chosen.sort(key=lambda i: rel[i])          # most similar last
        return [self.support[i] for i in chosen]
