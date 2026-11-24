"""Generate the bundled DEMO dataset (synthetic support-ticket corpus).

The corpus is SYNTHETIC: sentences are assembled from phrase banks with a
fixed seed. It exists so the application runs offline. It is NOT a benchmark
and results on it must not be reported as benchmark performance.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

LABELS = {
    "billing": {
        "openers": ["I was charged twice for", "My invoice for", "The refund for",
                    "I cannot find the receipt for", "There is a wrong amount on"],
        "topics": ["my subscription this month", "the annual plan", "my last payment",
                   "the premium upgrade", "my card renewal"],
        "closers": ["Please correct it.", "I need a credit back.", "Can someone check?",
                    "This is urgent for my accounts team.", "Please send a corrected bill."],
    },
    "technical": {
        "openers": ["The app crashes when I open", "Uploading a file to", "The export button in",
                    "Search results do not load in", "The dashboard freezes after"],
        "topics": ["the reports page", "the mobile version", "the data importer",
                   "the settings screen", "the sync feature"],
        "closers": ["It worked yesterday.", "The error message says timeout.",
                    "I tried reinstalling it.", "Screenshots are attached.", "Please help soon."],
    },
    "account_access": {
        "openers": ["I forgot the password for", "I am locked out of", "My two-factor code for",
                    "I cannot sign in to", "Please reset the login for"],
        "topics": ["my work account", "the admin portal", "my email login",
                   "the customer workspace", "the shared team profile"],
        "closers": ["The reset link never arrives.", "It says my account is disabled.",
                    "I changed phones recently.", "My manager needs access today.",
                    "Thanks in advance."],
    },
}

CLASS_SIZES = {"billing": 90, "technical": 60, "account_access": 50}


def build_corpus(seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for label, bank in LABELS.items():
        for _ in range(CLASS_SIZES[label]):
            text = f"{rng.choice(bank['openers'])} {rng.choice(bank['topics'])}. {rng.choice(bank['closers'])}"
            rows.append({"text": text, "label": label})
    df = pd.DataFrame(rows)
    n = len(df)
    for i in rng.sample(range(n), 6):
        df.loc[i, "text"] = None          # missing text
    for i in rng.sample(range(n), 4):
        df.loc[i, "label"] = None         # missing label
    dups = df.iloc[rng.sample(range(n), 8)].copy()
    dups["text"] = dups["text"].str.upper()  # case-variant duplicates
    df = pd.concat([df, dups], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "demo_support_tickets.csv"
    build_corpus().to_csv(out, index=False)
    print(f"wrote {out}")
