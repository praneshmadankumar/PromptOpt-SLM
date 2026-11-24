"""Configuration loading for the three execution profiles.

Settings live in config/profiles.json so that every experiment can be
reproduced from a single, version-controlled file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
PROFILE_NAMES = ("demo", "research", "full_benchmark")


class ConfigError(ValueError):
    """Raised when a profile is unknown or malformed."""


@dataclass(frozen=True)
class ProfileSettings:
    name: str
    backend: str
    model_name: str
    dataset: str
    shots_per_class: int
    val_per_class: int
    n_candidates: int
    halving_fractions: Tuple[float, ...]
    stability_lambda: float
    stability_permutations: int
    n_test: int
    seeds: Tuple[int, ...]
    max_forward_passes: int
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in ("simulated", "hf"):
            raise ConfigError(f"unknown backend '{self.backend}'")
        if self.shots_per_class < 1 or self.val_per_class < 2:
            raise ConfigError("shots_per_class >= 1 and val_per_class >= 2 required")
        if not self.halving_fractions or self.halving_fractions[-1] != 1.0:
            raise ConfigError("halving schedule must end at 1.0 (full validation set)")

    def to_dict(self) -> dict:
        return asdict(self)


def load_profile(name: str, path: Path | None = None) -> ProfileSettings:
    """Load a named execution profile from JSON."""
    if name not in PROFILE_NAMES:
        raise ConfigError(f"profile must be one of {PROFILE_NAMES}, got '{name}'")
    path = path or ROOT / "config" / "profiles.json"
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))[name]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load profile '{name}' from {path}: {exc}") from exc
    raw = dict(raw)
    raw["halving_fractions"] = tuple(raw["halving_fractions"])
    raw["seeds"] = tuple(raw["seeds"])
    return ProfileSettings(name=name, **raw)
