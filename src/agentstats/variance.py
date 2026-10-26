"""Variance components (ANOVA / expected mean squares), D-studies and run/task planning.

Model for the crossed design (system p, task t, run r nested in (p, t))::

    y_ptr = mu + a_p + b_t + c_pt + e_ptr
    Var(a_p) = s_p,  Var(b_t) = s_t,  Var(c_pt) = s_pt,  Var(e_ptr) = s_e

With n_p systems, n_t tasks and n_r runs per cell (balanced), the mean squares have expectations

    E[MS_e ] = s_e                                  (df = n_p n_t (n_r - 1))
    E[MS_pt] = s_e + n_r s_pt                       (df = (n_p - 1)(n_t - 1))
    E[MS_p ] = s_e + n_r s_pt + n_t n_r s_p         (df = n_p - 1)
    E[MS_t ] = s_e + n_r s_pt + n_p n_r s_t         (df = n_t - 1)

and the (unbiased, before truncation) estimators follow by solving:

    s_e  = MS_e
    s_pt = (MS_pt - MS_e) / n_r
    s_p  = (MS_p  - MS_pt) / (n_t n_r)
    s_t  = (MS_t  - MS_pt) / (n_p n_r)

On 0/1 scores this is a linear-probability decomposition: the components are variances of
cell-mean probabilities and of Bernoulli noise, not latent-scale effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._utils import check_estimand, check_int, check_level, z_two_sided
from .data import Tensor, build_tensor, prep

_COMPONENTS = ("p", "t", "pt", "e")


@dataclass(frozen=True)
class VarianceComponents:
    """Result of :func:`variance_components`.

    ``sigma2_*`` are the (truncated-at-0) estimates; ``raw`` holds the untruncated estimates,
    ``truncated`` flags which were negative and set to 0, ``ms`` the mean squares, and
    ``prop_*`` the shares of the sum of the truncated components.
    """

    sigma2_p: float
    sigma2_t: float
    sigma2_pt: float
    sigma2_e: float
    prop_p: float
    prop_t: float
    prop_pt: float
    prop_e: float
    n_p: int
    n_t: int
    n_r: int
    truncated: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    ms: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        names = {"p": "system", "t": "task", "pt": "system x task", "e": "run (residual)"}
        return pd.DataFrame(
            {
                "component": [names[k] for k in _COMPONENTS],
                "variance": [getattr(self, f"sigma2_{k}") for k in _COMPONENTS],
                "proportion": [getattr(self, f"prop_{k}") for k in _COMPONENTS],
                "truncated": [self.truncated[k] for k in _COMPONENTS],
            }
        )


def variance_components_array(y: np.ndarray) -> VarianceComponents:
    """EMS variance components for a balanced ``(n_p, n_t, n_r)`` array (no range checks).

    Estimand: components describe the population of systems, tasks and runs generating the data
    (random-effects model); they feed both estimands through :func:`d_study`.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 3:
        raise ValueError(f"expected a 3-d array (n_p, n_t, n_r), got shape {y.shape}")
    P, T, R = y.shape
    if P < 2 or T < 2 or R < 2:
        raise ValueError(
            f"need n_p >= 2, n_t >= 2 and n_r >= 2 to separate all components; got {y.shape}"
        )
    if not np.isfinite(y).all():
        raise ValueError("input contains NaN or infinite values")
    cm = y.mean(-1)  # (P, T) cell means
    mp = cm.mean(1)  # (P,)
    mt = cm.mean(0)  # (T,)
    gm = cm.mean()
    ss_p = T * R * ((mp - gm) ** 2).sum()
    ss_t = P * R * ((mt - gm) ** 2).sum()
    ss_pt = R * ((cm - mp[:, None] - mt[None, :] + gm) ** 2).sum()
    ss_e = ((y - cm[..., None]) ** 2).sum()
    ms_p = ss_p / (P - 1)
    ms_t = ss_t / (T - 1)
    ms_pt = ss_pt / ((P - 1) * (T - 1))
    ms_e = ss_e / (P * T * (R - 1))
    raw = {
        "e": ms_e,
        "pt": (ms_pt - ms_e) / R,
        "p": (ms_p - ms_pt) / (T * R),
        "t": (ms_t - ms_pt) / (P * R),
    }
    trunc = {k: bool(v < 0) for k, v in raw.items()}
    est = {k: max(float(v), 0.0) for k, v in raw.items()}
    total = sum(est.values())
    prop = {k: (est[k] / total if total > 0 else float("nan")) for k in est}
    return VarianceComponents(
        sigma2_p=est["p"],
        sigma2_t=est["t"],
        sigma2_pt=est["pt"],
        sigma2_e=est["e"],
        prop_p=prop["p"],
        prop_t=prop["t"],
        prop_pt=prop["pt"],
        prop_e=prop["e"],
        n_p=P,
        n_t=T,
        n_r=R,
        truncated=trunc,
        raw={k: float(v) for k, v in raw.items()},
        ms={"p": float(ms_p), "t": float(ms_t), "pt": float(ms_pt), "e": float(ms_e)},
    )


def variance_components(df: pd.DataFrame) -> VarianceComponents:
    """ANOVA expected-mean-squares variance components for the crossed p x t x r:(pt) design.

    Requires balanced input (every system has every task with the same number of runs); call
    :func:`agentstats.balance` first.  Negative estimates are truncated at 0 and flagged in
    ``truncated``.  See the module docstring for the estimators.  On 0/1 scores this is a
    linear-probability decomposition.  Estimand: population-level components (see
    :func:`d_study` for how they map to ``fixed_tasks`` / ``task_population`` quantities).
    """
    df = prep(df)
    t = build_tensor(df)
    if not t.balanced:
        raise ValueError(
            "variance_components needs a balanced design (every system x task cell present with "
            "the same number of runs); call balance(df, min_runs) first"
        )
    return variance_components_array(t.x)


@dataclass(frozen=True)
class VarianceComponentsSingle:
    """One-way random-effects result: tasks as the random factor, runs nested in tasks."""

    sigma2_task: float
    sigma2_run: float
    icc: float
    n_t: int
    n_r: int
    truncated_task: bool
    raw_sigma2_task: float
    ms_between: float
    ms_within: float


def variance_components_single(df: pd.DataFrame, system_id: str) -> VarianceComponentsSingle:
    """One-way random-effects ANOVA for one system: ``sigma2_task``, ``sigma2_run`` and ICC(1).

    ``sigma2_run = MS_within`` and ``sigma2_task = (MS_between - MS_within) / n_r`` (truncated at
    0 with a flag); ``ICC(1) = sigma2_task / (sigma2_task + sigma2_run)``.  Requires an equal
    number (>= 2) of runs on every task of the system and >= 2 tasks (use
    :func:`agentstats.balance`).  Estimand: population components; ``sigma2_task`` includes no
    run noise.  On 0/1 scores this is a linear-probability decomposition.
    """
    df = prep(df)
    if system_id not in set(df["system_id"]):
        raise ValueError(f"system_id {system_id!r} not found in data")
    t = build_tensor(df, systems=[system_id])
    if not t.balanced:
        raise ValueError(
            f"system {system_id!r} does not have the same number of runs on every task; "
            "call balance(df, min_runs, systems=[system_id]) first"
        )
    y = t.x[0]
    T, R = y.shape
    if T < 2 or R < 2:
        raise ValueError(f"need >= 2 tasks and >= 2 runs per task, got {T} tasks x {R} runs")
    cm = y.mean(1)
    ms_b = R * ((cm - cm.mean()) ** 2).sum() / (T - 1)
    ms_w = ((y - cm[:, None]) ** 2).sum() / (T * (R - 1))
    raw = (ms_b - ms_w) / R
    s_task = max(float(raw), 0.0)
    denom = s_task + ms_w
    icc = s_task / denom if denom > 0 else float("nan")
    return VarianceComponentsSingle(
        sigma2_task=s_task,
        sigma2_run=float(ms_w),
        icc=float(icc),
        n_t=T,
        n_r=R,
        truncated_task=bool(raw < 0),
        raw_sigma2_task=float(raw),
        ms_between=float(ms_b),
        ms_within=float(ms_w),
    )


@dataclass(frozen=True)
class DStudy:
    """Generalisability ``e_rho2``, dependability ``phi`` and fixed-task reliability ``fixed_tasks`` (arrays)."""

    e_rho2: np.ndarray
    phi: np.ndarray
    n_t: np.ndarray
    n_r: np.ndarray
    fixed_tasks: np.ndarray


def d_study(vc: VarianceComponents, n_t, n_r) -> DStudy:
    """Decision study: reliability of system rankings for ``n_t`` tasks and ``n_r`` runs per task.

    ``E rho2 = s_p / (s_p + s_pt/n_t + s_e/(n_t n_r))`` (relative decisions: ranking systems on
    the same tasks, so the task main effect ``s_t`` cancels) and
    ``Phi = s_p / (s_p + (s_t + s_pt)/n_t + s_e/(n_t n_r))`` (absolute decisions).  ``n_t`` and
    ``n_r`` may be scalars or arrays and are broadcast against each other.  Estimand: both
    coefficients treat tasks as a random sample (``task_population``).

    ``fixed_tasks`` is the fixed-task reliability of the ranking, the test-retest reliability of
    system scores when the same ``n_t`` tasks are rerun: the system-by-task interaction on those
    tasks is part of the true score, so
    ``s_p_fix = s_p + s_pt / n_t`` and ``fixed_tasks = s_p_fix / (s_p_fix + s_e / (n_t n_r))``.
    Estimand: ``fixed_tasks`` (it is never smaller than ``e_rho2``).
    """
    n_t, n_r = np.broadcast_arrays(np.asarray(n_t, dtype=float), np.asarray(n_r, dtype=float))
    if (n_t < 1).any() or (n_r < 1).any():
        raise ValueError("n_t and n_r must be >= 1")
    s_p, s_t, s_pt, s_e = vc.sigma2_p, vc.sigma2_t, vc.sigma2_pt, vc.sigma2_e
    rel_err = s_pt / n_t + s_e / (n_t * n_r)
    abs_err = (s_t + s_pt) / n_t + s_e / (n_t * n_r)
    if s_p + rel_err.min() <= 0:
        raise ValueError("all variance components are zero; reliability is undefined")
    s_p_fix = s_p + s_pt / n_t
    return DStudy(
        e_rho2=s_p / (s_p + rel_err),
        phi=s_p / (s_p + abs_err),
        n_t=n_t,
        n_r=n_r,
        fixed_tasks=s_p_fix / (s_p_fix + s_e / (n_t * n_r)),
    )


def trials_needed(
    df: pd.DataFrame,
    system_id: str,
    target_halfwidth: float,
    level: float = 0.95,
    estimand: str = "task_population",
    max_runs: int = 1000,
    n_t_grid=None,
):
    """Runs/tasks needed for a normal-approximation CI half-width <= ``target_halfwidth``.

    Variance components come from :func:`variance_components_single` (balanced input required).
    Half-width is ``z * sqrt(Var(mean))`` with ``Var = s_run / (n_t n_r)`` for ``fixed_tasks``
    and ``Var = s_task / n_t + s_run / (n_t n_r)`` for ``task_population``.

    * ``fixed_tasks``: ``n_t`` is the number of tasks in the data; returns the smallest integer
      ``n_r`` (int).
    * ``task_population``: returns a DataFrame with columns ``n_t, n_r, n_attempts, halfwidth``;
      for each ``n_t`` it gives the smallest ``n_r`` (rows needing ``n_r > max_runs`` are
      omitted).  Needs ``n_t > s_task / (target / z)^2``; a default grid of feasible task counts
      is used unless ``n_t_grid`` is given.

    Raises ``ValueError`` if the target is unreachable within ``max_runs``.
    """
    level = check_level(level)
    check_estimand(estimand)
    max_runs = check_int(max_runs, "max_runs")
    if not (target_halfwidth > 0 and np.isfinite(target_halfwidth)):
        raise ValueError("target_halfwidth must be a positive finite number")
    vc = variance_components_single(df, system_id)
    z = z_two_sided(level)
    v_target = (target_halfwidth / z) ** 2
    if estimand == "fixed_tasks":
        n_r = max(1, int(np.ceil(vc.sigma2_run / (vc.n_t * v_target) - 1e-12)))
        if n_r > max_runs:
            raise ValueError(
                f"fixed_tasks half-width {target_halfwidth} needs n_r={n_r} > max_runs={max_runs}"
            )
        return n_r
    s_task, s_run = vc.sigma2_task, vc.sigma2_run
    n_t_min = int(np.floor(s_task / v_target)) + 1  # smallest n_t with n_t * v_target > s_task
    if n_t_grid is None:
        mult = np.array([1, 1.1, 1.25, 1.5, 2, 3, 5, 10, 20, 50])
        grid = np.unique(np.maximum(np.ceil(n_t_min * mult), 1).astype(int))
    else:
        grid = np.unique(np.asarray(n_t_grid, dtype=int))
        if (grid < 1).any():
            raise ValueError("n_t_grid must contain positive integers")
    rows = []
    for n_t in grid:
        slack = n_t * v_target - s_task
        if slack <= 0:
            continue
        n_r = max(1, int(np.ceil(s_run / slack - 1e-12)))
        if n_r > max_runs:
            continue
        hw = z * np.sqrt(s_task / n_t + s_run / (n_t * n_r))
        rows.append((int(n_t), n_r, int(n_t) * n_r, float(hw)))
    if not rows:
        raise ValueError(
            f"task_population half-width {target_halfwidth} is not reachable with the given "
            f"n_t grid and max_runs={max_runs} (needs n_t >= {n_t_min})"
        )
    return pd.DataFrame(rows, columns=["n_t", "n_r", "n_attempts", "halfwidth"])
