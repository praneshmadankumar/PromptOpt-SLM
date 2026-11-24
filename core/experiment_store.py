"""SQLite-backed experiment log (Module 6): runs, prompt versions, metrics.

Schema (see report Chapter 3, ER diagram):
    runs(id, created_at, profile, dataset, backend, model_name, config_json, metrics_json)
    prompt_versions(id, run_id, signature, template_id, calibration, n_demos,
                    f1, std, J, config_json)
"""
from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    profile TEXT NOT NULL,
    dataset TEXT NOT NULL,
    backend TEXT NOT NULL,
    model_name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    metrics_json TEXT
);
CREATE TABLE IF NOT EXISTS prompt_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    signature TEXT NOT NULL,
    template_id TEXT,
    calibration TEXT,
    n_demos INTEGER,
    f1 REAL,
    std REAL,
    J REAL,
    config_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_pv_run ON prompt_versions(run_id);
"""


class ExperimentStore:
    """Thin, parameterised wrapper around sqlite3 (no string-built SQL)."""

    def __init__(self, path: str | Path = "results/experiments.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)

    def __enter__(self) -> "ExperimentStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def log_run(self, profile: str, dataset: str, backend: str, model_name: str,
                config: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO runs (created_at, profile, dataset, backend, model_name, config_json, metrics_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), profile, dataset, backend,
             model_name, json.dumps(config, sort_keys=True), json.dumps(metrics or {})))
        self._conn.commit()
        return int(cur.lastrowid)

    def update_metrics(self, run_id: int, metrics: Dict[str, Any]) -> None:
        self._conn.execute("UPDATE runs SET metrics_json = ? WHERE id = ?",
                           (json.dumps(metrics), run_id))
        self._conn.commit()

    def log_prompt_version(self, run_id: int, entry: Dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO prompt_versions (run_id, signature, template_id, calibration, n_demos, "
            "f1, std, J, config_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, entry["signature"], entry.get("template_id"), entry.get("calibration"),
             entry.get("n_demos"), entry.get("f1"), entry.get("std"), entry.get("J"),
             json.dumps(entry.get("config", {}), sort_keys=True)))
        self._conn.commit()

    def leaderboard(self, run_id: Optional[int] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        sql = ("SELECT run_id, signature, template_id, calibration, n_demos, f1, std, J "
               "FROM prompt_versions ")
        params: tuple = ()
        if run_id is not None:
            sql += "WHERE run_id = ? "
            params = (run_id,)
        sql += "ORDER BY J DESC LIMIT ?"
        params += (top_k,)
        cols = ["run_id", "signature", "template_id", "calibration", "n_demos", "f1", "std", "J"]
        return [dict(zip(cols, row)) for row in self._conn.execute(sql, params).fetchall()]

    def runs(self) -> List[Dict[str, Any]]:
        cols = ["id", "created_at", "profile", "dataset", "backend", "model_name"]
        rows = self._conn.execute("SELECT id, created_at, profile, dataset, backend, model_name "
                                  "FROM runs ORDER BY id DESC").fetchall()
        return [dict(zip(cols, r)) for r in rows]

    def export_csv(self, out: str | Path) -> Path:
        out = Path(out)
        rows = self.leaderboard(top_k=10_000)
        with out.open("w", newline="", encoding="utf-8") as fh:
            if rows:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        return out
