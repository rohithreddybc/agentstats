"""Paired comparison of two systems: bootstrap CI, permutation test, MDE and claim audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ._utils import as_rng, check_estimand, check_int, check_level
from .bootstrap import boot_system_means, check_resamplable, check_scheme
from .data import Tensor, build_tensor, prep

_EXACT_MAX_TASKS = 15
_TOL = 1e-12


@dataclass(frozen=True)
class PairedDiff:
    """Result of :func:`paired_diff` (``mean_diff = score(sys_a) - score(sys_b)``)."""

    system_a: str
    system_b: str
    mean_diff: float
    lo: float
    hi: float
    level: float
    p_value: float
    effect_size: float
    n_tasks: int
    estimand: str
    n_boot: int


def _shared_tensor(df: pd.DataFrame, sys_a: str, sys_b: str) -> Tensor:
    df = prep(df)
    if sys_a == sys_b:
        raise ValueError("sys_a and sys_b must differ")
    present = set(df["system_id"])
    for s in (sys_a, sys_b):
        if s not in present:
            raise ValueError(f"system_id {s!r} not found in data")
    sub = df[df["system_id"].isin([sys_a, sys_b])]
    ta = set(sub.loc[sub["system_id"] == sys_a, "task_id"])
    tb = set(sub.loc[sub["system_id"] == sys_b, "task_id"])
    shared = ta & tb
    if not shared:
        raise ValueError(f"systems {sys_a!r} and {sys_b!r} share no tasks")
    return build_tensor(sub, tasks=shared)


def _signflip_p(d: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    n = len(d)
    obs = abs(d.mean()) - _TOL
    if n <= _EXACT_MAX_TASKS:
        bits = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
        stat = np.abs((bits * 2.0 - 1.0) @ d) / n
        return float((stat >= obs).mean())
    count, done = 0, 0
    while done < n_perm:
        b = min(2000, n_perm - done)
        signs = rng.integers(0, 2, size=(b, n)) * 2.0 - 1.0
        count += int((np.abs(signs @ d) / n >= obs).sum())
        done += b
    return (1 + count) / (n_perm + 1)


def _run_perm_p(t: Tensor, ia: int, ib: int, n_perm: int, rng: np.random.Generator) -> float:
    """Run-level permutation test within tasks (labels of pooled runs shuffled per task)."""
    na, nb = t.counts[ia], t.counts[ib]
    Ra, Rb = int(na.max()), int(nb.max())
    T = len(na)
    M = Ra + Rb
    pooled = np.zeros((T, M))
    valid = np.zeros((T, M), dtype=bool)
    ja = np.arange(Ra)[None, :] < na[:, None]
    jb = np.arange(Rb)[None, :] < nb[:, None]
    pooled[:, :Ra] = t.x[ia][:, :Ra]
    pooled[:, Ra:] = t.x[ib][:, :Rb]
    valid[:, :Ra] = ja
    valid[:, Ra:] = jb
    tot = pooled.sum(1)
    first = np.arange(M)[None, :] < na[:, None]  # first n_a shuffled positions -> group A
    obs = abs((pooled[:, :Ra].sum(1) / na - pooled[:, Ra:].sum(1) / nb).mean()) - _TOL
    chunk = max(1, 2_000_000 // (T * M))
    count, done = 0, 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        keys = rng.random((b, T, M))
        keys[:, ~valid] = np.inf
        perm = np.argsort(keys, axis=-1)
        shuffled = np.take_along_axis(np.broadcast_to(pooled, (b, T, M)), perm, axis=-1)
        sa = (shuffled * first).sum(-1)  # (b, T)
        stat = np.abs((sa / na - (tot - sa) / nb).mean(-1))
        count += int((stat >= obs).sum())
        done += b
    return (1 + count) / (n_perm + 1)


def paired_diff(
    df: pd.DataFrame,
    sys_a: str,
    sys_b: str,
    level: float = 0.95,
    n_boot: int = 10000,
    estimand: str = "task_population",
    rng=None,
    scheme: str = "cluster",
) -> PairedDiff:
    """Paired difference ``score(sys_a) - score(sys_b)`` over the tasks both systems share.

    * Estimand ``task_population``: percentile CI from a paired bootstrap with shared task
      resamples (``scheme="cluster"``, default: each task keeps its runs; ``"two_level"``:
      runs also resampled within tasks, conservative); p-value from a task-level sign-flip permutation test on
      the per-task mean differences (exact enumeration when <= 15 tasks, otherwise ``n_boot``
      random sign vectors with the +1 correction).
    * Estimand ``fixed_tasks``: CI from adjusted residual resampling of runs within tasks
      (see :func:`agentstats.bootstrap_ci`; ``scheme`` ignored); p-value from a
      run-level permutation test in which the pooled runs of both systems are re-labelled at
      random within each task (``n_boot`` permutations, +1 correction).

    ``effect_size`` is the mean difference divided by the SD (ddof=1) of the task-level
    differences (NaN if that SD is 0).  Runs of the two systems are treated as unpaired.
    """
    level = check_level(level)
    n_boot = check_int(n_boot, "n_boot", 10)
    check_estimand(estimand)
    check_scheme(scheme)
    rng = as_rng(rng)
    t = _shared_tensor(df, sys_a, sys_b)
    ia, ib = t.systems.index(sys_a), t.systems.index(sys_b)
    T = t.x.shape[1]
    if T < 2:
        raise ValueError("need at least 2 shared tasks")
    check_resamplable(t, estimand)
    cm = t.cell_means()
    d = cm[ia] - cm[ib]
    mean_diff = float(d.mean())
    reps = boot_system_means(t, n_boot, estimand, rng, scheme)
    rd = reps[:, ia] - reps[:, ib]
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(rd, [a, 1.0 - a])
    if estimand == "task_population":
        p = _signflip_p(d, n_boot, rng)
    else:
        p = _run_perm_p(t, ia, ib, n_boot, rng)
    sd = d.std(ddof=1)
    eff = mean_diff / sd if sd > 0 else float("nan")
    return PairedDiff(sys_a, sys_b, mean_diff, float(lo), float(hi), level, float(p), float(eff),
                      int(T), estimand, n_boot)


# ---------------------------------------------------------------------------------------------
# Minimum detectable effect
# ---------------------------------------------------------------------------------------------


def _mde_var(sigma2_delta_task, sigma2_run, n_t, n_r, estimand):
    s_d = np.asarray(sigma2_delta_task, dtype=float)
    s_r = np.asarray(sigma2_run, dtype=float)
    n_t = np.asarray(n_t, dtype=float)
    n_r = np.asarray(n_r, dtype=float)
    if (s_d < 0).any() or (s_r < 0).any():
        raise ValueError("variances must be >= 0")
    if (n_t < 1).any() or (n_r < 1).any():
        raise ValueError("n_t and n_r must be >= 1")
    if estimand == "task_population":
        return (s_d + 2.0 * s_r / n_r) / n_t
    return 2.0 * s_r / (n_t * n_r)


def mde(
    sigma2_delta_task,
    sigma2_run,
    n_t,
    n_r,
    alpha: float = 0.05,
    power: float = 0.8,
    estimand: str = "task_population",
):
    """Minimum detectable paired difference between two systems (two-sided test).

    ``MDE = (z_{1-alpha/2} + z_{power}) * sqrt(Var)`` where, for ``task_population``,
    ``Var = (sigma2_delta_task + 2 sigma2_run / n_r) / n_t`` and, for ``fixed_tasks``,
    ``Var = 2 sigma2_run / (n_t n_r)`` (``sigma2_delta_task`` is ignored: the task set is fixed).
    ``sigma2_delta_task`` is the variance across tasks of the *true* per-task difference;
    ``sigma2_run`` the run-to-run variance of one system (assumed equal for both).
    Vectorised over all numeric arguments.
    """
    check_estimand(estimand)
    check_level(alpha, "alpha")
    check_level(power, "power")
    var = _mde_var(sigma2_delta_task, sigma2_run, n_t, n_r, estimand)
    k = stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power)
    out = k * np.sqrt(var)
    return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class MDEResult:
    """Result of :func:`mde_from_df`."""

    mde: float
    sigma2_delta_task: float
    sigma2_run: float
    n_t: int
    n_r: int
    alpha: float
    power: float
    estimand: str


def mde_from_df(
    df: pd.DataFrame,
    sys_a: str,
    sys_b: str,
    n_t=None,
    n_r=None,
    alpha: float = 0.05,
    power: float = 0.8,
    estimand: str = "task_population",
) -> MDEResult:
    """Estimate the variances from data and return the MDE between two systems.

    ``sigma2_run`` is the average of the two systems' mean within-cell variances (cells with
    >= 2 runs).  ``sigma2_delta_task`` is ``Var_t(d_t)`` (``d_t`` = per-task difference of cell
    means) minus the run noise it contains, ``mean_t(s2_a/n_a + s2_b/n_b)``, truncated at 0.
    ``n_t`` defaults to the number of shared tasks and ``n_r`` to the smallest cell run count
    over both systems; pass values to plan a different design.
    """
    check_estimand(estimand)
    t = _shared_tensor(df, sys_a, sys_b)
    ia, ib = t.systems.index(sys_a), t.systems.index(sys_b)
    T = t.x.shape[1]
    if T < 2:
        raise ValueError("need at least 2 shared tasks")
    cnt = t.counts.astype(float)
    cm = t.cell_means()
    s2 = np.full(cnt.shape, np.nan)
    ok = cnt >= 2
    ss = (t.x**2).sum(-1) - cnt * cm**2
    s2[ok] = np.maximum(ss[ok], 0.0) / (cnt[ok] - 1)
    pooled = np.array([np.nanmean(s2[i]) if ok[i].any() else np.nan for i in range(2)])
    if np.isnan(pooled[[ia, ib]]).any():
        raise ValueError("each system needs at least one task with >= 2 runs to estimate sigma2_run")
    s2f = np.where(ok, s2, pooled[:, None])
    d = cm[ia] - cm[ib]
    noise = np.mean(s2f[ia] / cnt[ia] + s2f[ib] / cnt[ib])
    s_delta = max(float(d.var(ddof=1) - noise), 0.0)
    s_run = float(pooled[[ia, ib]].mean())
    n_t_ = T if n_t is None else check_int(n_t, "n_t")
    n_r_ = int(cnt[[ia, ib]].min()) if n_r is None else check_int(n_r, "n_r")
    m = mde(s_delta, s_run, n_t_, n_r_, alpha, power, estimand)
    return MDEResult(m, s_delta, s_run, int(n_t_), int(n_r_), alpha, power, estimand)


# ---------------------------------------------------------------------------------------------
# Published-claim audit
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimCheck:
    """Result of :func:`claim_check`."""

    diff: float
    se: float
    z: float
    p_value: float
    mde: float
    verdict: str


def claim_check(
    reported_a: float,
    reported_b: float,
    n_t: int,
    n_r: int,
    sigma2_delta_task: float,
    sigma2_run: float,
    alpha: float = 0.05,
    estimand: str = "task_population",
    power: float = 0.8,
    clear_alpha: float = 0.005,
) -> ClaimCheck:
    """Audit a reported gap using only published scores plus borrowed variance components.

    ``z = (a - b) / sqrt(Var)`` with ``Var`` as in :func:`mde`, ``p`` the two-sided normal
    p-value, and ``mde`` the minimum detectable difference at ``power`` (a property of the design,
    reported next to the result; it never enters the verdict).  The verdict describes the
    evidence only, from ``p``:

    * ``"clear"``: ``p < clear_alpha`` (default 0.005);
    * ``"marginal"``: ``clear_alpha <= p < alpha``;
    * ``"not distinguishable"``: ``p >= alpha``.

    There is no "underpowered" verdict: classifying a result by its observed effect against the
    MDE is the post-hoc power fallacy (observed power is a function of ``p``).  Compare the
    reported gap with ``mde`` as a statement about the design, not about the result.

    Estimand: ``task_population`` treats the tasks as a sample; ``fixed_tasks`` counts only
    run-to-run noise.  The variance components are borrowed, so the result is only as good as
    that borrowing.
    """
    check_estimand(estimand)
    check_level(alpha, "alpha")
    check_level(power, "power")
    check_level(clear_alpha, "clear_alpha")
    if clear_alpha > alpha:
        raise ValueError("clear_alpha must be <= alpha")
    if not (np.isfinite(reported_a) and np.isfinite(reported_b)):
        raise ValueError("reported scores must be finite")
    var = float(_mde_var(sigma2_delta_task, sigma2_run, n_t, n_r, estimand))
    if var <= 0:
        raise ValueError("variance of the difference is 0; supply positive variance components")
    se = float(np.sqrt(var))
    diff = float(reported_a - reported_b)
    z = diff / se
    p = float(2.0 * stats.norm.sf(abs(z)))
    m = mde(sigma2_delta_task, sigma2_run, n_t, n_r, alpha, power, estimand)
    if p < clear_alpha:
        verdict = "clear"
    elif p < alpha:
        verdict = "marginal"
    else:
        verdict = "not distinguishable"
    return ClaimCheck(diff, se, z, p, float(m), verdict)
