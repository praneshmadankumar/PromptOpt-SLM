"""Reusable Streamlit/Plotly helpers for the dashboard."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

PROVENANCE_BANNER = {
    "simulated": "DEMO MODE: the SimulatedBackend is a lexicon heuristic, NOT a language model. "
                 "Outputs are plumbing checks, not SLM results.",
    "hf": "RESEARCH MODE: real Hugging Face model. Metrics are valid only for the configured budget.",
}


def probability_frame(labels: Sequence[str], probs: Sequence[float]) -> pd.DataFrame:
    """Tidy table for a bar chart of class probabilities."""
    return pd.DataFrame({"class": list(labels), "probability": [float(p) for p in probs]})


def leaderboard_frame(rows: Sequence[dict]) -> pd.DataFrame:
    cols = ["rank", "signature", "template_id", "calibration", "stage", "score", "J", "f1", "std"]
    df = pd.DataFrame(list(rows))
    return df[[c for c in cols if c in df.columns]] if len(df) else df
