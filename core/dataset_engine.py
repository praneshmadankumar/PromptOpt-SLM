"""Module 1 - Dataset Intelligence Engine.

Loads CSV/JSON corpora, validates and cleans them, reports quality metrics,
and produces stratified support / validation / test splits plus few-shot
samples.

Quality score (0-100) used throughout the report:
    Q = 100 * (1 - missing_rate) * (1 - duplicate_rate) * balance
    balance = min_class_count / max_class_count
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

_WS = re.compile(r"\s+")


class DatasetError(ValueError):
    """Raised when a dataset cannot be loaded, validated or split."""


def normalise_text(text: object) -> str:
    """Collapse whitespace and strip; returns '' for missing values."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return _WS.sub(" ", str(text)).strip()


@dataclass
class QualityReport:
    n_rows_raw: int
    n_missing_text: int
    n_missing_label: int
    n_duplicates: int
    class_counts: Dict[str, int]
    imbalance_ratio: float
    length_mean: float
    length_median: float
    length_p95: float
    quality_score: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class DatasetEngine:
    """Container for one labelled text-classification corpus."""

    def __init__(self, df: pd.DataFrame, text_col: str = "text", label_col: str = "label"):
        missing = [c for c in (text_col, label_col) if c not in df.columns]
        if missing:
            raise DatasetError(f"missing required column(s) {missing}; found {list(df.columns)}")
        self._raw = (df[[text_col, label_col]]
                     .rename(columns={text_col: "text", label_col: "label"})
                     .copy())
        self._clean: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------ loaders
    @classmethod
    def from_csv(cls, path: str | Path, text_col: str = "text", label_col: str = "label") -> "DatasetEngine":
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise DatasetError(f"cannot read CSV '{path}': {exc}") from exc
        return cls(df, text_col, label_col)

    @classmethod
    def from_json(cls, path: str | Path, text_col: str = "text", label_col: str = "label") -> "DatasetEngine":
        try:
            if hasattr(path, "read"):                      # file-like (e.g. Streamlit upload)
                raw = path.read()
                raw = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            else:
                raw = Path(path).read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetError(f"cannot read JSON '{path}': {exc}") from exc
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if not isinstance(data, list):
            raise DatasetError("JSON corpus must be a list of records (or {'data': [...]})")
        return cls(pd.DataFrame(data), text_col, label_col)

    # ------------------------------------------------------------ cleaning
    def quality_report(self) -> QualityReport:
        raw = self._raw
        n = len(raw)
        if n == 0:
            raise DatasetError("dataset is empty")
        miss_text = int(raw["text"].isna().sum())
        miss_label = int(raw["label"].isna().sum())
        complete = raw.dropna(subset=["label"]).copy()
        complete["_key"] = complete["text"].map(normalise_text).str.lower()
        n_dup = int(complete.duplicated(subset=["_key"]).sum())
        usable = complete[complete["_key"] != ""].drop_duplicates(subset=["_key"])
        counts = usable["label"].astype(str).value_counts()
        class_counts = {str(k): int(v) for k, v in counts.items()}
        imbalance = float(counts.min() / counts.max()) if len(counts) else 0.0
        lengths = usable["_key"].str.split().str.len().to_numpy()
        missing_rate = (miss_text + miss_label) / (2 * n)
        dup_rate = n_dup / max(1, n)
        score = 100.0 * (1 - missing_rate) * (1 - dup_rate) * imbalance
        notes = []
        if miss_text or miss_label:
            notes.append(f"{miss_text} missing texts and {miss_label} missing labels will be dropped.")
        if n_dup:
            notes.append(f"{n_dup} duplicate texts (case/whitespace-insensitive) will be removed.")
        if imbalance < 0.5:
            notes.append("Severe class imbalance: consider class-balanced demonstrations.")
        return QualityReport(
            n_rows_raw=n, n_missing_text=miss_text, n_missing_label=miss_label,
            n_duplicates=n_dup, class_counts=class_counts,
            imbalance_ratio=round(imbalance, 4),
            length_mean=float(lengths.mean()) if len(lengths) else 0.0,
            length_median=float(np.median(lengths)) if len(lengths) else 0.0,
            length_p95=float(np.percentile(lengths, 95)) if len(lengths) else 0.0,
            quality_score=round(score, 2), notes=notes,
        )

    def clean(self) -> "DatasetEngine":
        """Normalise text, drop missing rows, remove duplicates. Returns self."""
        df = self._raw.dropna(subset=["label"]).copy()
        df["text"] = df["text"].map(normalise_text)
        df = df[df["text"] != ""]
        df["_key"] = df["text"].str.lower()
        df = df.drop_duplicates(subset=["_key"]).drop(columns="_key")
        df["label"] = df["label"].astype(str)
        self._clean = df.reset_index(drop=True)
        if self._clean.empty:
            raise DatasetError("no usable rows after cleaning")
        return self

    @property
    def data(self) -> pd.DataFrame:
        if self._clean is None:
            self.clean()
        return self._clean

    @property
    def classes(self) -> List[str]:
        return sorted(self.data["label"].unique().tolist())

    # ------------------------------------------------------------ splitting
    def split(self, val_per_class: int, test_fraction: float = 0.3, seed: int = 0) -> Dict[str, pd.DataFrame]:
        """Stratified support/validation/test split.

        Returns a dict with keys 'pool' (for few-shot support sampling),
        'val' (prompt selection only) and 'test' (final reporting only).
        The test set is never touched by any search component.
        """
        df = self.data
        counts = df["label"].value_counts()
        if (counts < val_per_class + 2).any():
            raise DatasetError("some classes are too small for the requested validation size")
        pool_df, test_df = train_test_split(df, test_size=test_fraction, stratify=df["label"],
                                            random_state=seed)
        # validation: exactly val_per_class examples per class, drawn from the non-test pool
        val_parts = []
        for _, grp in pool_df.groupby("label"):
            val_parts.append(grp.sample(n=val_per_class, random_state=seed))
        val_df = pd.concat(val_parts)
        support_pool = pool_df.drop(val_df.index)
        return {"pool": support_pool.reset_index(drop=True),
                "val": val_df.reset_index(drop=True),
                "test": test_df.reset_index(drop=True)}

    def few_shot_sample(self, pool: pd.DataFrame, shots_per_class: int, seed: int) -> List[tuple]:
        """Class-balanced random sample of (text, class_index) pairs."""
        classes = self.classes
        rng = np.random.default_rng(seed)
        chosen = []
        for cid, label in enumerate(classes):
            grp = pool[pool["label"] == label]
            if len(grp) < shots_per_class:
                raise DatasetError(f"class '{label}' has only {len(grp)} pool examples")
            idx = rng.choice(len(grp), size=shots_per_class, replace=False)
            chosen += [(grp.iloc[i]["text"], cid) for i in idx]
        return chosen

    # ------------------------------------------------------------ analysis
    def semantic_similarity_summary(self, n_sample: int = 200, seed: int = 0) -> dict:
        """Mean pairwise TF-IDF cosine similarity within and across classes."""
        df = self.data
        sample = df.sample(n=min(n_sample, len(df)), random_state=seed)
        X = TfidfVectorizer().fit_transform(sample["text"])
        sim = (X @ X.T).toarray()
        labels = np.asarray(sample["label"], dtype=object)
        same = labels[:, None] == labels[None, :]
        off = ~np.eye(len(sample), dtype=bool)
        return {
            "mean_within_class": float(sim[same & off].mean()) if (same & off).any() else 0.0,
            "mean_between_class": float(sim[~same].mean()) if (~same).any() else 0.0,
        }
