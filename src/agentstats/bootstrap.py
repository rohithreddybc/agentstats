"""Vectorised bootstrap engine, bootstrap CIs and the interquartile mean.

Estimands (stated in every public docstring):

* ``fixed_tasks`` - uncertainty of the score on THIS benchmark's task set; runs are random.
  Resampling: adjusted residual resampling of runs within each cell (see below).
* ``task_population`` - uncertainty about the population the tasks were drawn from; tasks and
  runs both random.  Two schemes:

  - ``scheme="cluster"`` (default): resample tasks with replacement and keep each task's runs
    intact.  The variance of the resampled task means already contains the run noise, so nothing
    is double counted.
  - ``scheme="two_level"``: resample tasks, then runs within each sampled task.  Conservative:
    the run noise is counted twice (Davison & Hinkley 1997, sec. 3.8).

``fixed_tasks`` resamples, within each (system, task) cell with ``R_c`` runs, the centred
residuals and inflates them by ``sqrt(R_c / (R_c - 1))`` (Davison & Hinkley 1997, sec. 3.3):
``score* = cell_mean + sqrt(R_c/(R_c-1)) * (resampled_score - cell_mean)``.  This removes the
``(R_c - 1)/R_c`` variance deficit of plain resampling.  Cells with ``R_c = 1`` contribute zero
variance (a warning is issued when more than 10% of cells have a single run).

All resampling is done with index arrays (no Python loop over resamples); the only Python loop is
over memory-bounded chunks of resamples.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._utils import as_rng, check_estimand, check_int, check_level
from .data import Tensor, build_tensor, prep

_MAX_ELEMS = 6_000_000  # per-chunk size of the (b, P, T, R) index arrays
SCHEMES = ("cluster", "two_level")


@dataclass(frozen=True)
class CI:
    """Percentile bootstrap interval for a point estimate ``mean``."""

    mean: float
    lo: float
    hi: float
    level: float
    estimand: str
    n_boot: int

    @property
    def halfwidth(self) -> float:
        return (self.hi - self.lo) / 2.0


def check_scheme(scheme: str) -> str:
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    return scheme


def check_resamplable(t: Tensor, estimand: str) -> None:
    """Raise if the resampling implied by ``estimand`` cannot produce any variation."""
    if not t.complete:
        raise ValueError(
            "every system must have at least one run on every task; use balance() or restrict "
            "to common tasks first"
        )
    n_tasks = t.counts.shape[1]
    if estimand == "task_population" and n_tasks < 2:
        raise ValueError("task_population bootstrap needs at least 2 tasks")
    if estimand == "fixed_tasks":
        if t.counts.max() < 2:
            raise ValueError(
                "fixed_tasks bootstrap needs at least 2 runs on some task (all cells have 1 run, "
                "so run-to-run variance is not identifiable)"
            )
        frac = float((t.counts == 1).mean())
        if frac > 0.10:
            warnings.warn(
                f"{100 * frac:.0f}% of (system, task) cells have a single run and contribute zero "
                "run-to-run variance to the fixed_tasks bootstrap; the interval is too narrow",
                UserWarning,
                stacklevel=3,
            )


def boot_chunks(
    t: Tensor, n_boot: int, mode: str, rng: np.random.Generator, adjust_values: bool = True
):
    """Yield ``(vals, cnt, cell)`` chunks of bootstrap resamples.

    ``mode``: ``"cluster"`` (tasks resampled, runs kept), ``"two_level"`` (tasks, then runs
    resampled with replacement) or ``"runs"`` (task set fixed, adjusted residual resampling of
    runs).  ``vals`` has shape ``(b, P, T, R)``; ``cnt`` is ``(b, P, T)`` (or the scalar ``R``
    when balanced) and gives the valid draws per cell; padded positions of ragged cells hold 0
    so that ``vals.sum(-1) / cnt`` is the resampled cell mean.  Task resamples are shared across
    systems (paired); run resamples are independent across systems and cells.  ``cell`` is the
    ``(b, P, T)`` array of flat cell ids.  In ``"runs"`` mode the residual adjustment is applied to
    the values unless ``adjust_values=False`` (the caller then adjusts the cell means with
    ``m + k * (m_star - m)``, which is much cheaper).
    """
    P, T, R = t.x.shape
    xflat = t.x.ravel()
    cflat = t.counts.ravel()
    balanced = bool((t.counts == R).all())
    chunk = max(1, _MAX_ELEMS // (P * T * R))
    p_off = (np.arange(P) * T)[None, :, None]
    if mode == "runs":
        with np.errstate(invalid="ignore", divide="ignore"):
            mflat = (t.x.sum(-1) / t.counts).ravel()
            kflat = np.where(cflat > 1, np.sqrt(cflat / np.maximum(cflat - 1, 1)), 0.0)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        if mode == "two_level" or mode == "cluster":
            tcell = rng.integers(0, T, size=(b, T))
        else:
            tcell = np.broadcast_to(np.arange(T), (b, T))
        cell = p_off + tcell[:, None, :]  # (b, P, T) flat cell ids
        if mode == "cluster":
            vals = xflat.reshape(P * T, R)[cell]
            cnt = R if balanced else cflat[cell]
            yield vals, cnt, cell
            done += b
            continue
        base = (cell * R)[..., None]
        if balanced:
            ridx = rng.integers(0, R, size=(b, P, T, R), dtype=np.int32)
            cnt = R
            idx = ridx + base
            valid = None
        else:
            cnt_c = cflat[cell]  # (b, P, T)
            u = rng.random((b, P, T, R), dtype=np.float32)
            u *= cnt_c[..., None].astype(np.float32)
            ridx = u.astype(np.int64)
            np.minimum(ridx, (cnt_c - 1)[..., None], out=ridx)
            valid = np.arange(R) < cnt_c[..., None]
            idx = ridx + base
            cnt = cnt_c
        vals = xflat.take(idx)
        if mode == "runs" and adjust_values:
            m = mflat[cell][..., None]
            vals = m + kflat[cell][..., None] * (vals - m)
        if valid is not None:
            vals *= valid
        yield vals, cnt, cell
        done += b


def boot_system_means(
    t: Tensor, n_boot: int, estimand: str, rng: np.random.Generator, scheme: str = "cluster"
) -> np.ndarray:
    """Bootstrap replicates of every system's mean score, shape ``(n_boot, P)``.

    The mean is the mean over tasks of the (resampled) per-task run mean.  ``scheme`` only
    matters for ``task_population``.
    """
    check_resamplable(t, estimand)
    check_scheme(scheme)
    P, T, _ = t.x.shape
    out = np.empty((n_boot, P))
    if estimand == "task_population" and scheme == "cluster":
        cm = t.cell_means()  # (P, T)
        chunk = max(1, _MAX_ELEMS // (P * T))
        s = 0
        while s < n_boot:
            b = min(chunk, n_boot - s)
            tidx = rng.integers(0, T, size=(b, T))
            out[s : s + b] = cm[:, tidx].mean(-1).T
            s += b
        return out
    mode = "two_level" if estimand == "task_population" else "runs"
    if mode == "runs":
        with np.errstate(invalid="ignore", divide="ignore"):
            mflat = (t.x.sum(-1) / t.counts).ravel()
        cflat = t.counts.ravel()
        kflat = np.where(cflat > 1, np.sqrt(cflat / np.maximum(cflat - 1, 1)), 0.0)
    s = 0
    for vals, cnt, cell in boot_chunks(t, n_boot, mode, rng, adjust_values=False):
        b = vals.shape[0]
        cm = vals.sum(-1) / cnt
        if mode == "runs":
            m = mflat[cell]
            cm = m + kflat[cell] * (cm - m)
        out[s : s + b] = cm.mean(-1)
        s += b
    return out


def _percentile_ci(reps: np.ndarray, point: float, level: float, estimand: str) -> CI:
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(reps, [a, 1.0 - a])
    return CI(float(point), float(lo), float(hi), level, estimand, int(len(reps)))


def _system_tensor(df: pd.DataFrame, system_id: str) -> Tensor:
    df = prep(df)
    if system_id not in set(df["system_id"]):
        raise ValueError(f"system_id {system_id!r} not found in data")
    return build_tensor(df, systems=[system_id])


def bootstrap_ci(
    df: pd.DataFrame,
    system_id: str,
    level: float = 0.95,
    n_boot: int = 10000,
    estimand: str = "task_population",
    rng=None,
    scheme: str = "cluster",
) -> CI:
    """Percentile bootstrap CI for one system's mean score.

    Estimand ``task_population`` (default): tasks resampled with replacement.  With
    ``scheme="cluster"`` (default) each sampled task keeps its runs, which already carry the
    run noise (no double counting).  ``scheme="two_level"`` additionally resamples runs within
    each sampled task; this counts the run noise twice and is conservative (variance over-stated
    by about ``sigma2_run (n_r - 1) / n_r**2 / n_t``).

    Estimand ``fixed_tasks``: the task set is fixed and runs are resampled within each task by
    adjusted residual resampling (centred residuals inflated by ``sqrt(R_c/(R_c-1))``), which
    corrects the ``(R_c-1)/R_c`` variance deficit of plain run resampling; ``scheme`` is
    ignored.  Cells with a single run contribute no variance (warning if > 10% of cells).

    The point estimate is the mean over tasks of the per-task mean score.
    """
    level = check_level(level)
    n_boot = check_int(n_boot, "n_boot", 10)
    check_estimand(estimand)
    check_scheme(scheme)
    rng = as_rng(rng)
    t = _system_tensor(df, system_id)
    reps = boot_system_means(t, n_boot, estimand, rng, scheme)[:, 0]
    point = t.cell_means().mean()
    return _percentile_ci(reps, point, level, estimand)


# ---------------------------------------------------------------------------------------------
# Interquartile mean
# ---------------------------------------------------------------------------------------------


def _iqm_last_axis(a: np.ndarray) -> np.ndarray:
    n = a.shape[-1]
    s = np.sort(a, axis=-1)
    lo, hi = n / 4.0, 3.0 * n / 4.0
    i = np.arange(n)
    w = np.clip(np.minimum(i + 1.0, hi) - np.maximum(i, lo), 0.0, None)
    return (s * w).sum(-1) / (hi - lo)


def iqm(x) -> float:
    """Interquartile mean: mean of the middle 50% of the values.

    Exact for any sample size: values straddling the 25th/75th percentile boundary get
    fractional weight (so for ``n`` divisible by 4 it equals ``scipy.stats.trim_mean(x, 0.25)``).
    Not an inferential statistic, so no estimand applies.
    """
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("iqm of an empty array is undefined")
    if not np.isfinite(a).all():
        raise ValueError("iqm input contains NaN or infinite values")
    return float(_iqm_last_axis(a))


def iqm_ci(
    df: pd.DataFrame,
    system_id: str,
    level: float = 0.95,
    n_boot: int = 2000,
    estimand: str = "task_population",
    rng=None,
    scheme: str = "cluster",
) -> CI:
    """Stratified-bootstrap CI for the IQM of one system's per-attempt scores (rliable style).

    The statistic is the IQM of all ``n_tasks * n_runs`` attempt scores pooled.  Estimand
    ``task_population`` (default): tasks resampled with replacement; ``scheme="cluster"`` keeps
    each task's runs, ``scheme="two_level"`` also resamples runs within tasks (conservative).
    Estimand ``fixed_tasks``: stratified by task, runs resampled within each task by adjusted
    residual resampling (see :func:`bootstrap_ci`); this is the rliable-style stratified
    bootstrap with the small-``R`` variance correction.  Requires a balanced system (equal runs
    per task); use :func:`agentstats.balance` first.  For binary 0/1 scores the IQM is a
    degenerate summary.
    """
    level = check_level(level)
    n_boot = check_int(n_boot, "n_boot", 10)
    check_estimand(estimand)
    check_scheme(scheme)
    rng = as_rng(rng)
    t = _system_tensor(df, system_id)
    if not t.balanced:
        raise ValueError("iqm_ci requires equal run counts per task; call balance() first")
    check_resamplable(t, estimand)
    _, T, R = t.x.shape
    mode = "runs" if estimand == "fixed_tasks" else scheme
    reps = np.empty(n_boot)
    s = 0
    for vals, _, _ in boot_chunks(t, n_boot, mode, rng):
        b = vals.shape[0]
        reps[s : s + b] = _iqm_last_axis(vals.reshape(b, T * R))
        s += b
    return _percentile_ci(reps, iqm(t.x), level, estimand)
