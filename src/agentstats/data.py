"""Data schema, validation, balancing and basic summaries.

Long format, one row per attempt: ``benchmark, task_id, system_id, run_id, score`` (+ optional
columns). A *cell* is one (system, task) pair; its ``run_id`` rows are the repeated runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._utils import check_int

REQUIRED_COLUMNS = ("benchmark", "task_id", "system_id", "run_id", "score")
_STR_COLUMNS = ("benchmark", "task_id", "system_id")
_OPT_STR = ("model", "harness", "source")
_OPT_NUM = ("cost_usd", "wall_s", "tokens_in", "tokens_out")
KEY = ["benchmark", "task_id", "system_id", "run_id"]


def _is_str_series(s: pd.Series) -> bool:
    return pd.api.types.infer_dtype(s, skipna=False) == "string"


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check the long-format schema and return a (possibly dtype-normalised) frame.

    Raises ``ValueError`` on: non-DataFrame/empty input, missing required columns, non-string
    ``benchmark``/``task_id``/``system_id``, non-integer ``run_id``, non-numeric / NaN /
    out-of-[0, 1] ``score``, non-numeric optional numeric columns, and duplicate
    (benchmark, task_id, system_id, run_id) keys.  Returned frame has ``run_id`` as int64 and
    ``score`` as float64.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"expected a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required column(s): {missing}")
    if len(df) == 0:
        raise ValueError("data frame is empty")
    for c in _STR_COLUMNS:
        if not _is_str_series(df[c]):
            raise ValueError(f"column {c!r} must contain only str values (got dtype {df[c].dtype})")
    for c in _OPT_STR:
        if c in df.columns:
            nn = df[c].dropna()
            if len(nn) and not _is_str_series(nn):
                raise ValueError(f"column {c!r} must contain only str values")
    run = df["run_id"]
    if pd.api.types.is_bool_dtype(run) or not pd.api.types.is_integer_dtype(run):
        raise ValueError(f"column 'run_id' must have an integer dtype (got {run.dtype})")
    if run.isna().any():
        raise ValueError("column 'run_id' contains missing values")
    sc = df["score"]
    if not pd.api.types.is_numeric_dtype(sc) and not pd.api.types.is_bool_dtype(sc):
        raise ValueError(f"column 'score' must be numeric (got dtype {sc.dtype})")
    scv = sc.to_numpy(dtype=float)
    if not np.isfinite(scv).all():
        raise ValueError("column 'score' contains NaN or infinite values")
    if (scv < 0).any() or (scv > 1).any():
        raise ValueError(
            f"column 'score' must lie in [0, 1]; found range [{scv.min():.6g}, {scv.max():.6g}]"
        )
    for c in _OPT_NUM:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"column {c!r} must be numeric (got dtype {df[c].dtype})")
    dup = df.duplicated(subset=KEY, keep=False)
    if dup.any():
        ex = df.loc[dup, KEY].head(3).to_dict("records")
        raise ValueError(
            f"{int(dup.sum())} rows share a duplicate (benchmark, task_id, system_id, run_id) "
            f"key, e.g. {ex}"
        )
    if run.dtype != np.int64 or sc.dtype != np.float64:
        df = df.assign(run_id=run.astype(np.int64), score=scv)
    return df


def prep(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and require a single benchmark (used by every public function)."""
    df = validate(df)
    nb = df["benchmark"].nunique()
    if nb > 1:
        raise ValueError(
            f"data contains {nb} benchmarks {sorted(df['benchmark'].unique())[:5]}; "
            "filter to one benchmark first"
        )
    return df


@dataclass
class Tensor:
    """Padded (system, task, run) score array.

    ``x[p, t, :counts[p, t]]`` are the scores of cell (p, t) in run_id order; the rest is zero
    padding.  ``counts[p, t] == 0`` means the cell is absent.
    """

    x: np.ndarray
    counts: np.ndarray
    systems: list
    tasks: list

    @property
    def complete(self) -> bool:
        return bool((self.counts > 0).all())

    @property
    def balanced(self) -> bool:
        return self.complete and bool((self.counts == self.counts.flat[0]).all())

    def cell_means(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.x.sum(-1) / self.counts


def build_tensor(df: pd.DataFrame, systems=None, tasks=None) -> Tensor:
    """Pivot a validated single-benchmark frame to a padded :class:`Tensor` (sorted ids)."""
    if systems is not None:
        df = df[df["system_id"].isin(list(systems))]
    if tasks is not None:
        df = df[df["task_id"].isin(list(tasks))]
    if len(df) == 0:
        raise ValueError("no rows left after selecting systems/tasks")
    sc, su = pd.factorize(df["system_id"], sort=True)
    tc, tu = pd.factorize(df["task_id"], sort=True)
    P, T = len(su), len(tu)
    order = np.lexsort((df["run_id"].to_numpy(), tc, sc))
    sc, tc = sc[order], tc[order]
    score = df["score"].to_numpy(dtype=float)[order]
    cell = sc * T + tc
    counts = np.bincount(cell, minlength=P * T)
    starts = np.cumsum(counts) - counts
    pos = np.arange(len(cell)) - starts[cell]
    x = np.zeros((P * T, int(counts.max())))
    x[cell, pos] = score
    return Tensor(x.reshape(P, T, -1), counts.reshape(P, T), list(su), list(tu))


def balance(df: pd.DataFrame, min_runs: int, systems=None):
    """Make a (system x task) design with exactly ``min_runs`` runs per cell.

    Keeps tasks on which *every* selected system has >= ``min_runs`` runs (a missing cell counts
    as 0 runs) and truncates each kept cell to its first ``min_runs`` ``run_id`` values.  If
    ``systems`` is given only those systems are returned.

    Returns ``(balanced_df, info)``; ``info`` has ``min_runs``, ``systems``, ``n_tasks_total``,
    ``n_tasks_kept``, ``n_tasks_dropped``, ``dropped_tasks``, ``rows_in``, ``rows_out``,
    ``rows_dropped_with_tasks`` and ``rows_truncated``.  Raises ``ValueError`` if no task
    survives.
    """
    df = prep(df)
    min_runs = check_int(min_runs, "min_runs")
    all_sys = sorted(df["system_id"].unique())
    if systems is None:
        systems = all_sys
    else:
        systems = sorted(set(systems))
        unknown = [s for s in systems if s not in set(all_sys)]
        if unknown or not systems:
            raise ValueError(f"unknown or empty systems selection: {unknown or systems}")
    sub = df[df["system_id"].isin(systems)]
    counts = (
        sub.groupby(["task_id", "system_id"]).size().unstack("system_id").reindex(columns=systems)
    )
    ok = (counts.fillna(0) >= min_runs).all(axis=1)
    kept = sorted(ok.index[ok])
    dropped = sorted(ok.index[~ok])
    if not kept:
        raise ValueError(
            f"no task has >= {min_runs} runs for every one of the {len(systems)} selected systems"
        )
    rows_in = len(sub)
    sub = sub[sub["task_id"].isin(kept)].sort_values(["system_id", "task_id", "run_id"])
    rows_after_tasks = len(sub)
    pos = sub.groupby(["system_id", "task_id"]).cumcount().to_numpy()
    out = sub[pos < min_runs].reset_index(drop=True)
    info = {
        "min_runs": min_runs,
        "systems": list(systems),
        "n_tasks_total": int(len(ok)),
        "n_tasks_kept": len(kept),
        "n_tasks_dropped": len(dropped),
        "dropped_tasks": dropped,
        "rows_in": int(rows_in),
        "rows_out": int(len(out)),
        "rows_dropped_with_tasks": int(rows_in - rows_after_tasks),
        "rows_truncated": int(rows_after_tasks - len(out)),
    }
    return out, info


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean score per (system, task) over runs, plus ``n_runs``.

    Estimand: none (descriptive); the cell means are the task-level units used by the
    ``task_population`` estimand.
    """
    df = prep(df)
    g = df.groupby(["benchmark", "system_id", "task_id"], sort=True)["score"]
    return g.agg(mean="mean", n_runs="size").reset_index()


def system_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-system score: mean over tasks of the per-task mean (each task weighted equally).

    Columns: ``system_id, mean, n_tasks, n_runs_min, n_runs_max``.  Estimand: this is the point
    estimate for both ``fixed_tasks`` (score on this task set) and ``task_population``.
    """
    cm = cell_means(df)
    g = cm.groupby("system_id", sort=True)
    return pd.DataFrame(
        {
            "mean": g["mean"].mean(),
            "n_tasks": g.size(),
            "n_runs_min": g["n_runs"].min(),
            "n_runs_max": g["n_runs"].max(),
        }
    ).reset_index()
