"""Small shared helpers: RNG coercion and argument validation."""

from __future__ import annotations

import numpy as np
from scipy import stats

ESTIMANDS = ("fixed_tasks", "task_population")


def as_rng(rng) -> np.random.Generator:
    """Coerce ``rng`` (Generator | int | None) to a ``numpy.random.Generator``."""
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (bool, np.bool_)):
        raise ValueError("rng must be a numpy Generator, an int seed or None, not bool")
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    raise ValueError(
        f"rng must be a numpy Generator, an int seed or None, got {type(rng).__name__}"
    )


def check_estimand(estimand: str) -> str:
    if estimand not in ESTIMANDS:
        raise ValueError(f"estimand must be one of {ESTIMANDS}, got {estimand!r}")
    return estimand


def check_level(level: float, name: str = "level") -> float:
    if not (isinstance(level, (int, float, np.floating)) and 0.0 < level < 1.0):
        raise ValueError(f"{name} must be in (0, 1), got {level!r}")
    return float(level)


def check_int(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def z_two_sided(level: float) -> float:
    """Normal critical value for a two-sided interval of coverage ``level``."""
    return float(stats.norm.ppf(0.5 + level / 2.0))
