"""The paper's named quantities as first-class functions.

* Q1 :func:`run_disagreement`: run disagreement rate ``D`` and run share ``rho_run``.
* Q3 :func:`reversal_probability`, :func:`pair_z`, :func:`pair_z_table`: probability that the
  order of two systems flips on a rerun.
* Q4 :func:`runs_needed`, :func:`fit_se_curve`: runs needed to separate a gap, and the
  run-variance curve ``s_e(p) = kappa p (1 - p)`` it is evaluated on.  The design MDE is
  :func:`agentstats.mde`; fixed-task reliability is the ``fixed_tasks`` output of
  :func:`agentstats.d_study`.
* :func:`plan_second_stage`: a two-stage adaptive design.  After a first pass of a few runs per
  cell, it spends the rest of a fixed per-system budget on the cells whose outcomes disagree
  (Neyman allocation on an add-smoothed estimate of the run SD).

Estimands.  ``fixed_tasks``: the task set is the object of inference, only runs are random.
``task_population``: tasks are a sample from a population, tasks and runs are both random.
Each docstring states which one it targets.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from ._utils import as_rng, check_estimand, check_int, check_level
from .compare import _shared_tensor
from .data import Tensor, build_tensor, prep

__all__ = [
    "PairZ",
    "RunDisagreement",
    "RunsNeeded",
    "SECurve",
    "fit_se_curve",
    "pair_z",
    "pair_z_table",
    "plan_second_stage",
    "reversal_probability",
    "run_disagreement",
    "runs_needed",
]


# ---------------------------------------------------------------------------------------------
# Q1: run disagreement
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SECurve:
    """Fit of ``s_e(p) = kappa * p * (1 - p)`` (least squares through the origin).

    ``kappa < 1`` when cell success probabilities differ across tasks (Jensen:
    ``mean_t p_t (1 - p_t) = m (1 - m) - Var_t p_t``).  Calling the object, ``curve(p)``, gives
    ``s_e`` at score ``p``.
    """

    kappa: float
    kappa_se: float
    r2: float
    n_systems: int

    def __call__(self, p):
        p = np.asarray(p, dtype=float)
        out = self.kappa * p * (1.0 - p)
        return float(out) if out.ndim == 0 else out

    def s_e_band(self, p, z: float = 1.0):
        """``(lo, hi)`` of ``s_e(p)`` with ``kappa -+ z * kappa_se`` (lower kappa floored at 1e-9)."""
        p = np.asarray(p, dtype=float)
        q = p * (1.0 - p)
        lo = np.maximum(self.kappa - z * self.kappa_se, 1e-9) * q
        hi = (self.kappa + z * self.kappa_se) * q
        return (float(lo), float(hi)) if q.ndim == 0 else (lo, hi)


def fit_se_curve(m, s_e, w=None) -> SECurve:
    """Fit ``s_e = kappa * m * (1 - m)`` across systems by weighted least squares through the origin.

    ``m`` are system mean scores, ``s_e`` the per-system run variances (for example the
    ``by_system`` table of :func:`run_disagreement`), ``w`` optional weights (the analysis weights
    by the number of cells behind each ``s_e``).  Non-finite pairs are dropped.  ``kappa_se`` is
    the usual through-the-origin standard error ``sqrt(sigma2 / sum(w q^2))`` with
    ``sigma2 = sum(w resid^2) / max(n - 1, 1)``; ``r2`` is ``1 - SSE / SST`` about the mean of
    ``s_e``.  Estimand: a descriptive fit across the systems supplied (``fixed_tasks``).
    """
    m, s_e = np.asarray(m, dtype=float), np.asarray(s_e, dtype=float)
    if m.shape != s_e.shape or m.ndim != 1:
        raise ValueError("m and s_e must be 1-d arrays of equal length")
    ok = np.isfinite(m) & np.isfinite(s_e)
    m, s_e = m[ok], s_e[ok]
    if w is None:
        w = np.ones_like(m)
    else:
        w = np.asarray(w, dtype=float)
        if w.shape != ok.shape:
            raise ValueError("w must have the same length as m")
        w = w[ok]
    if len(m) == 0:
        raise ValueError("no finite (m, s_e) pairs to fit")
    q = m * (1.0 - m)
    den = float((w * q * q).sum())
    kappa = float((w * q * s_e).sum() / den) if den > 0 else float("nan")
    resid = s_e - kappa * q
    n = len(m)
    sigma2 = float((w * resid**2).sum() / max(n - 1, 1))
    se_kappa = float(np.sqrt(sigma2 / den)) if den > 0 else float("nan")
    ss_tot = float(((s_e - s_e.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return SECurve(kappa=kappa, kappa_se=se_kappa, r2=r2, n_systems=int(n))


@dataclass(frozen=True)
class RunDisagreement:
    """Result of :func:`run_disagreement`.

    ``D`` is the release-level run disagreement rate (``= 2 * s_e``), ``s_e`` the pooled unbiased
    run variance, ``s_pt`` the system x task interaction variance used in ``rho_run``
    (``s_pt_method`` says how it was obtained), ``rho_run = s_e / (s_pt + s_e)``.
    ``by_system`` has one row per system: ``system_id, D, s_e, mean, n_tasks, n_cells_used``
    (``mean`` weights tasks equally).
    """

    D: float
    s_e: float
    s_pt: float
    rho_run: float
    s_pt_method: str
    n_cells: int
    n_systems: int
    n_tasks: int
    by_system: pd.DataFrame = field(repr=False)

    def se_curve(self) -> SECurve:
        """:func:`fit_se_curve` of the per-system ``s_e`` on the system mean, weighted by cells used."""
        b = self.by_system
        return fit_se_curve(b["mean"].to_numpy(), b["s_e"].to_numpy(), b["n_cells_used"].to_numpy(float))


def _cell_s2(t: Tensor) -> np.ndarray:
    """Unbiased within-cell variance for a padded tensor; NaN where fewer than 2 runs."""
    n = t.counts.astype(float)
    cm = t.cell_means()
    ss = (t.x**2).sum(-1) - n * np.where(n > 0, cm, 0.0) ** 2
    ok = n >= 2
    out = np.full(n.shape, np.nan)
    out[ok] = np.maximum(ss[ok], 0.0) / (n[ok] - 1.0)
    return out


def run_disagreement(df: pd.DataFrame, s_pt: float | None = None) -> RunDisagreement:
    """Run disagreement rate ``D`` and run share ``rho_run`` for one release, with per-system ``D``.

    Estimand: ``fixed_tasks``.  ``D`` is the probability that two distinct runs of the same
    (system, task) cell differ in outcome, averaged over the observed cells.  The unit is a
    cell with ``R_c >= 2`` runs and ``k_c`` successes (cells with one run are skipped), and the
    unbiased cell estimator is ``D_c = 2 k_c (R_c - k_c) / (R_c (R_c - 1))`` (valid for ragged
    ``R_c``).  The release-level ``D`` is the mean of ``D_c`` over cells (equal weight per cell,
    across all systems), which equals ``2 * s_e``; the per-system ``D`` is the same mean within
    one system.  For non-binary scores the code generalises ``D_c`` to ``2 *`` the unbiased
    cell variance, i.e. the mean squared difference of two runs, which reduces to the formula
    above on 0/1 scores.

    ``rho_run = s_e / (s_pt + s_e)`` is the share of cell-level variation due to run noise in
    the crossed model ``y = mu + a_p + b_t + c_pt + e``.  ``s_e`` is the ragged estimator above.
    ``s_pt`` is estimated from the unweighted cell means ``m_pt`` (needs every system x task
    cell present, >= 2 systems and >= 2 tasks)::

        s_pt = max(MS_pt* - mean_c(s2_c / R_c), 0)

    where ``MS_pt*`` is the interaction mean square of the two-way table of cell means, with
    ``(P - 1)(T - 1)`` degrees of freedom, and ``s2_c`` is the unbiased within-cell run variance
    of cell ``c`` (``R_c`` runs).  With heterogeneous cell variances ``sigma2_c`` (binary
    outcomes), ``E[MS_pt*] = s_pt + mean_c(sigma2_c / R_c)`` exactly, so subtracting
    ``mean_c(s2_c / R_c)`` is unbiased before truncation.  The earlier form
    ``s_e * mean_c(1 / R_c)`` (product of means) is biased when cell variance and run count are
    associated (fixed in 0.2.1).  A cell with a single run has no ``s2_c``; its term uses the
    pooled ``s_e``, so the correction is exact only when every cell has >= 2 runs.  On balanced
    data (equal ``R_c >= 2``) the two forms coincide and the estimate equals the ANOVA estimator
    of :func:`variance_components`.  On an incomplete design ``s_pt`` and
    ``rho_run`` are NaN (with a warning) while ``D`` is unaffected.  Pass ``s_pt`` to use a
    value computed elsewhere (for example from a balanced subset); ``s_pt_method`` then reads
    ``"supplied"``.

    Intervals are not computed here: resample tasks (task-cluster bootstrap) around this
    function.
    """
    df = prep(df)
    t = build_tensor(df)
    s2 = _cell_s2(t)
    ok = np.isfinite(s2)
    if not ok.any():
        raise ValueError("need at least one (system, task) cell with >= 2 runs")
    s_e = float(s2[ok].mean())
    cm = t.cell_means()
    rows = []
    for i, sid in enumerate(t.systems):
        present = t.counts[i] > 0
        oki = ok[i]
        se_i = float(s2[i][oki].mean()) if oki.any() else float("nan")
        rows.append(
            dict(
                system_id=sid,
                D=2.0 * se_i,
                s_e=se_i,
                mean=float(cm[i][present].mean()),
                n_tasks=int(present.sum()),
                n_cells_used=int(oki.sum()),
            )
        )
    by_system = pd.DataFrame(rows)

    P, T = t.counts.shape
    if s_pt is not None:
        if not (np.isfinite(s_pt) and s_pt >= 0):
            raise ValueError("s_pt must be a finite number >= 0")
        spt, method = float(s_pt), "supplied"
    elif t.complete and P >= 2 and T >= 2:
        mp, mt, gm = cm.mean(1), cm.mean(0), cm.mean()
        ms_pt = ((cm - mp[:, None] - mt[None, :] + gm) ** 2).sum() / ((P - 1) * (T - 1))
        s2f = np.where(ok, s2, s_e)  # single-run cells: pooled s_e stands in for s2_c
        spt = max(float(ms_pt - (s2f / t.counts).mean()), 0.0)
        method = "unweighted cell-mean interaction MS minus mean(s2_c / R_c), truncated at 0"
    else:
        spt, method = float("nan"), "unavailable: incomplete system x task design"
        warnings.warn(
            "run_disagreement: system x task design is incomplete (or has < 2 systems/tasks), so "
            "s_pt and rho_run are NaN; D is unaffected. Restrict to common tasks or pass s_pt.",
            stacklevel=2,
        )
    denom = spt + s_e
    rho = s_e / denom if np.isfinite(denom) and denom > 0 else float("nan")
    return RunDisagreement(
        D=2.0 * s_e,
        s_e=s_e,
        s_pt=spt,
        rho_run=float(rho),
        s_pt_method=method,
        n_cells=int(ok.sum()),
        n_systems=P,
        n_tasks=T,
        by_system=by_system,
    )


# ---------------------------------------------------------------------------------------------
# Q3: reversal probability
# ---------------------------------------------------------------------------------------------


def reversal_probability(z):
    """Predicted probability that a rerun reverses the order of a pair: ``Phi(-|z| / sqrt(2))``.

    ``z = gap / SE_diff`` for the gap seen in one evaluation, with ``SE_diff`` the fixed-task
    standard error of that gap (see :func:`pair_z`).  Goodman (1992): the replication
    difference has standard error ``sqrt(2) * SE_diff`` under a flat prior on the true gap.
    Estimand: ``fixed_tasks`` (rerun the same tasks).  Vectorised; NaN in, NaN out.
    """
    z = np.asarray(z, dtype=float)
    out = stats.norm.cdf(-np.abs(z) / np.sqrt(2.0))
    return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class PairZ:
    """Result of :func:`pair_z` (``gap = score(system_a) - score(system_b)`` over shared tasks)."""

    system_a: str
    system_b: str
    gap: float
    se_diff: float
    z: float
    psi: float
    n_tasks: int


def _pair_row(t: Tensor, s2: np.ndarray, ia: int, ib: int, s_e: Mapping | float | None) -> PairZ:
    both = (t.counts[ia] > 0) & (t.counts[ib] > 0)
    T = int(both.sum())
    if T < 1:
        raise ValueError(f"systems {t.systems[ia]!r} and {t.systems[ib]!r} share no tasks")
    cm = t.cell_means()
    gap = float((cm[ia][both] - cm[ib][both]).mean())
    parts = []
    for i in (ia, ib):
        sid = t.systems[i]
        if s_e is None:
            v = s2[i][both]
            fin = np.isfinite(v)
            if not fin.any():
                raise ValueError(
                    f"system {sid!r} has no shared task with >= 2 runs to estimate s_e; pass s_e"
                )
            # per-cell unbiased variance; cells with a single run take the system's mean s2
            s2_cells = np.where(fin, v, v[fin].mean())
        else:
            if isinstance(s_e, Mapping):
                if sid not in s_e:
                    raise ValueError(f"s_e has no entry for system {sid!r}")
                se_i = float(s_e[sid])
            else:
                se_i = float(s_e)
            if not (np.isfinite(se_i) and se_i >= 0):
                raise ValueError("s_e must be finite and >= 0")
            s2_cells = np.full(int(both.sum()), se_i)
        # sum_t s2_t / R_t (not mean(s2) * mean(1/R): biased for unequal R, see pair_z)
        parts.append(float((s2_cells / t.counts[i][both]).sum()) / T)
    se_diff = float(np.sqrt(sum(parts) / T))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = gap / se_diff if se_diff > 0 else (0.0 if gap == 0 else float("inf") * np.sign(gap))
    return PairZ(t.systems[ia], t.systems[ib], gap, se_diff, float(z), reversal_probability(z), T)


def pair_z(df: pd.DataFrame, sys_a: str, sys_b: str, s_e: Mapping | float | None = None) -> PairZ:
    """Pair-level ``z`` of the gap between two systems, with the fixed-task standard error.

    Estimand: ``fixed_tasks``.  Over the ``T`` tasks both systems share,
    ``gap = mean_t(m_a,t - m_b,t)`` (cell means) and

        SE_diff^2 = T^-2 * sum_t ( s2_a,t / R_a,t + s2_b,t / R_b,t )

    where ``s2_a,t`` is the unbiased run variance of the cell (system ``a``, task ``t``) and
    ``R_a,t`` its number of runs.  This is unbiased for the fixed-task variance of the gap even
    when the runs per cell differ and the cell variances are heterogeneous (binary outcomes);
    versions before 0.2.1 used ``mean_t(s2) * mean_t(1 / R)``, which is biased in that case.
    It equals ``(s_e,a + s_e,b) / (T k)`` when every cell has ``k`` runs, with ``s_e,a`` the
    mean cell variance, so balanced designs are unchanged.  A shared cell with a single run has
    no ``s2``; it takes the system's mean ``s2`` over its cells with >= 2 runs.  If ``s_e`` is
    given (a float for both systems, or a mapping ``system_id -> s_e``) it replaces every
    ``s2_a,t`` of that system; this is required when a system has no cell with 2 runs, for
    example a single-run evaluation using the full-data ``s_e``.  ``z = gap / SE_diff`` is signed; ``psi = reversal_probability(z)``.
    """
    if sys_a == sys_b:
        raise ValueError("sys_a and sys_b must differ")
    t = _shared_tensor(df, sys_a, sys_b)
    return _pair_row(t, _cell_s2(t), t.systems.index(sys_a), t.systems.index(sys_b), s_e)


def pair_z_table(df: pd.DataFrame, s_e: Mapping | float | None = None) -> pd.DataFrame:
    """:func:`pair_z` for every pair of systems (rows ordered by system id): one row per pair.

    Columns ``system_a, system_b, gap, se_diff, z, psi, n_tasks``.  Pairs whose standard error
    cannot be estimated raise ``ValueError`` as in :func:`pair_z`.
    """
    df = prep(df)
    t = build_tensor(df)
    s2 = _cell_s2(t)
    rows = []
    for ia in range(len(t.systems)):
        for ib in range(ia + 1, len(t.systems)):
            r = _pair_row(t, s2, ia, ib, s_e)
            rows.append(r.__dict__)
    if not rows:
        raise ValueError("need at least 2 systems")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Q4: runs needed
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunsNeeded:
    """Result of :func:`runs_needed`.  ``n_r`` is ``None`` when ``determinable`` is False.

    ``z`` is the multiplier used: ``z_{1-alpha/2} + z_{power}`` for ``criterion="power"`` and
    ``z_{1-risk}`` for ``criterion="reversal"``.  ``risk`` is only used by the reversal criterion
    and ``alpha``/``power`` only by the power criterion.
    """

    n_r: int | None
    n_r_cont: float
    determinable: bool
    d: float
    n_t: float
    s_e: float
    risk: float
    estimand: str
    criterion: str = "power"
    alpha: float = 0.05
    power: float = 0.8
    z: float = float("nan")


def runs_needed(
    d: float,
    s_e: float | None,
    n_t: float,
    risk: float = 0.05,
    estimand: str = "fixed_tasks",
    sigma2_delta_task: float | None = None,
    *,
    p: float | None = None,
    curve: SECurve | None = None,
    criterion: str = "power",
    alpha: float = 0.05,
    power: float = 0.8,
) -> RunsNeeded:
    """Smallest integer number of runs per task ``n_r`` so that a true gap ``d`` is resolved.

    ``criterion="power"`` (default, the headline ``n_r*(d)``): the design MDE of the two-sided
    test at level ``alpha`` with power ``power`` is at most ``d``,
    ``(z_{1-alpha/2} + z_{power}) SE_diff <= d`` (the same MDE as :func:`agentstats.mde`).
    ``criterion="reversal"`` (secondary, ``n_r_rev``): a rerun reverses a true gap ``d`` with
    probability at most ``risk``, ``Phi(-d / SE_diff) <= risk``, i.e. ``z_{1-risk} SE_diff <= d``.
    The reversal criterion is the weaker one: at ``alpha = 0.05``, ``power = 0.8`` the power
    multiplier is 2.80 against 1.645 for ``risk = 0.05``, so ``n_r`` (power) is about 2.9 times
    ``n_r_rev``.  ``risk`` is ignored by the power criterion, ``alpha`` and ``power`` by the
    reversal criterion.

    ``d`` is the gap in score units (``0.02`` for two points).  ``n_t`` is the number of tasks.
    ``s_e`` is the run variance of one system (assumed equal for both); instead of ``s_e`` pass
    the score level ``p`` together with a fitted ``curve`` (:func:`fit_se_curve`), which uses
    ``s_e = curve(p) = kappa p (1 - p)``.

    * ``fixed_tasks`` (default): ``SE_diff^2 = 2 s_e / (n_t n_r)``, so
      ``n_r = ceil(2 s_e z^2 / (n_t d^2))``, at least 1.
    * ``task_population``: ``SE_diff^2 = (sigma2_delta_task + 2 s_e / n_r) / n_t``; needs
      ``sigma2_delta_task`` (the variance across tasks of the true per-task difference).  If
      ``n_t d^2 / z^2 <= sigma2_delta_task`` no number of runs suffices: ``determinable`` is
      False and ``n_r`` is ``None`` (more tasks are needed, not more runs).

    ``n_r_cont`` is the unrounded value.  For an interval, call again with the band of
    ``s_e`` (``curve.s_e_band``).
    """
    check_estimand(estimand)
    if criterion not in ("power", "reversal"):
        raise ValueError(f"criterion must be 'power' or 'reversal', got {criterion!r}")
    if not (np.isfinite(d) and d > 0):
        raise ValueError("d must be a positive finite number")
    if not (np.isfinite(n_t) and n_t >= 1):
        raise ValueError("n_t must be >= 1")
    if criterion == "reversal":
        if not (0.0 < risk < 0.5):
            raise ValueError(f"risk must be in (0, 0.5), got {risk!r}")
        z = float(stats.norm.ppf(1.0 - risk))
    else:
        check_level(alpha, "alpha")
        check_level(power, "power")
        z = float(stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power))
    if s_e is None:
        if p is None or curve is None:
            raise ValueError("pass s_e, or both p and curve")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        s_e = curve(p)
    if not (np.isfinite(s_e) and s_e >= 0):
        raise ValueError("s_e must be finite and >= 0")
    meta = dict(criterion=criterion, alpha=float(alpha), power=float(power), z=z)
    if estimand == "fixed_tasks":
        cont = 2.0 * s_e * z**2 / (n_t * d**2)
    else:
        if sigma2_delta_task is None or not (np.isfinite(sigma2_delta_task) and sigma2_delta_task >= 0):
            raise ValueError("task_population needs sigma2_delta_task >= 0")
        slack = n_t * d**2 / z**2 - sigma2_delta_task
        if slack <= 0:
            return RunsNeeded(None, float("inf"), False, float(d), float(n_t), float(s_e), risk, estimand, **meta)
        cont = 2.0 * s_e / slack
    n_r = max(1, int(np.ceil(cont - 1e-12)))
    return RunsNeeded(n_r, float(cont), True, float(d), float(n_t), float(s_e), risk, estimand, **meta)


# ---------------------------------------------------------------------------------------------
# Two-stage adaptive allocation
# ---------------------------------------------------------------------------------------------


def _neyman_integer(weights, lo, hi, total, rng=None) -> np.ndarray:
    """Integer Neyman allocation along the last axis, vectorised over the leading axes.

    Minimises ``sum_t w_t^2 / n_t`` subject to ``sum_t n_t = total`` and ``lo_t <= n_t <= hi_t``.
    The continuous optimum is ``n_t = clip(lam * w_t, lo_t, hi_t)`` with ``lam`` found by bisection
    (the sum is non-decreasing in ``lam``).  The continuous solution is rounded down and the
    remaining runs are handed out one each by systematic random rounding: cell ``t`` gets the extra
    run with probability equal to its fractional part (so ``E[n_t]`` is the continuous optimum),
    with the row total exact and the cells visited in random order.  (Rounding to the largest
    fractional parts instead would erase differences smaller than one run.)
    ``weights`` must be positive; ``lo``, ``hi`` broadcast to ``weights``; ``total`` broadcasts to
    its leading axes and must lie in ``[sum lo, sum hi]``.  Returns an integer array of the shape
    of ``weights`` whose rows sum to ``total``.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim == 0:
        raise ValueError("weights must have at least one axis")
    if not (np.isfinite(w).all() and (w > 0).all()):
        raise ValueError("weights must be finite and > 0")
    lo = np.broadcast_to(np.asarray(lo, dtype=float), w.shape)
    hi = np.broadcast_to(np.asarray(hi, dtype=float), w.shape)
    tot = np.broadcast_to(np.asarray(total, dtype=float), w.shape[:-1])
    if (lo > hi).any():
        raise ValueError("lo must not exceed hi")
    if (tot < lo.sum(-1) - 1e-9).any() or (tot > hi.sum(-1) + 1e-9).any():
        raise ValueError("total must lie between sum(lo) and sum(hi)")
    rng = as_rng(rng)
    lam_lo = np.zeros(w.shape[:-1])
    lam_hi = (hi / w).max(-1)  # every cell is at hi here, so the sum is sum(hi) >= total
    for _ in range(50):
        mid = 0.5 * (lam_lo + lam_hi)
        small = np.clip(mid[..., None] * w, lo, hi).sum(-1) < tot
        lam_lo = np.where(small, mid, lam_lo)
        lam_hi = np.where(small, lam_hi, mid)
    n = np.clip(0.5 * (lam_lo + lam_hi)[..., None] * w, lo, hi)
    base = np.floor(n)
    frac = np.where(base < hi, n - base, 0.0)
    rest = tot - base.sum(-1)  # runs still to hand out, one each
    fs = frac.sum(-1)
    frac = frac * (np.where(fs > 0, rest / np.where(fs > 0, fs, 1.0), 0.0))[..., None]  # sum(frac) == rest exactly
    # systematic (dependent) rounding: cell t gets one extra run with probability frac_t, the row total is exact,
    # and the cells are visited in random order so no task is favoured by its position
    order = np.argsort(rng.random(w.shape), axis=-1)
    cum = np.cumsum(np.take_along_axis(frac, order, axis=-1), axis=-1)
    u = rng.random(w.shape[:-1])[..., None]
    pick = np.floor(cum + u) - np.floor(cum - np.take_along_axis(frac, order, axis=-1) + u)
    add = np.zeros(w.shape)
    np.put_along_axis(add, order, np.clip(pick, 0, 1), axis=-1)
    return (base + add).astype(np.int64)


def plan_second_stage(
    df_stage1: pd.DataFrame,
    budget,
    floor: int = 0,
    r_max: int | None = None,
    smoothing: float = 0.5,
    rng=None,
) -> pd.DataFrame:
    """Runs per (system, task) for the second stage of a two-stage adaptive design.

    Estimand: ``fixed_tasks``.  Stage 1 is a first pass with ``n1`` runs per cell (``df_stage1``,
    long format, ragged ``n1`` allowed).  The system's mean score is the mean over tasks of the
    cell means, whose variance is ``T^-2 sum_t s2_t / n_t`` (cell run variance ``s2_t``, ``n_t``
    runs in total); for a fixed total ``sum_t n_t`` it is smallest for ``n_t`` proportional to the
    cell run SD (Neyman allocation).  The cell SD is estimated from stage 1 with add-``smoothing``
    smoothing, ``p = (k + a) / (n1 + 2a)`` and ``sd = sqrt(p (1 - p))`` for ``k`` the summed stage-1
    score, so cells whose first runs agree keep a positive weight (``a = 0.5`` is
    ``(k + 0.5) / (n1 + 1)``).  The scores are treated as success indicators; for scores strictly
    inside ``(0, 1)`` the weight is an approximation.

    ``budget`` is the total number of runs per system, stage 1 included (an int for every
    system, or a mapping ``system_id -> int``).  Each cell gets at least ``n1 + floor`` runs
    (``floor`` extra runs for every cell, 0 by default) and at most ``r_max`` (default: no cap).
    The continuous allocation is rounded so each system's ``n_total`` sums exactly to its budget;
    rounding ties are broken at random with ``rng``.  When ``budget`` exceeds ``r_max`` times the
    number of cells, every cell gets ``r_max`` and a warning is issued.

    Returns one row per cell: ``system_id, task_id, n_stage1, n_stage2, n_total, p_smoothed,
    weight``.

    Caution: the allocation depends on the stage-1 outcomes, so the plain cell means of an
    adaptive design are biased estimates of the cell means, and the ragged fixed-task variance of
    :func:`pair_z` (which assumes run counts fixed in advance) under-covers.  On resampled public
    releases (``n1 = 2``, three runs per task) the 95% interval covered the full-data score in only
    0.69 to 0.88 of repetitions, the bias of a system's score reached about one percentage point
    with the default settings (about three with other tunings), and the design saved no runs once
    bias was counted (mean squared error) or when runs were drawn fresh.  Use uniform runs per task unless a run-variance estimate for each task exists that does
    not come from the system's own pilot runs (for example from other systems or earlier
    releases); such an allocation keeps the cell means unbiased.  This function is kept for
    experimentation, not as a recommended design.
    """
    df = prep(df_stage1)
    floor = check_int(floor, "floor", 0)
    if r_max is not None:
        r_max = check_int(r_max, "r_max", 1)
    if not (np.isfinite(smoothing) and smoothing > 0):
        raise ValueError(f"smoothing must be a positive finite number, got {smoothing!r}")
    rng = as_rng(rng)
    g = df.groupby(["system_id", "task_id"], sort=True)["score"].agg(["size", "sum"]).reset_index()
    out = []
    for sid, sub in g.groupby("system_id", sort=True):
        if isinstance(budget, Mapping):
            if sid not in budget:
                raise ValueError(f"budget has no entry for system {sid!r}")
            b = check_int(budget[sid], f"budget[{sid!r}]", 1)
        else:
            b = check_int(budget, "budget", 1)
        n1 = sub["size"].to_numpy(dtype=np.int64)
        k = sub["sum"].to_numpy(dtype=float)
        lo = n1 + floor
        hi = np.full(len(n1), b if r_max is None else r_max, dtype=np.int64)
        if (lo > hi).any():
            raise ValueError(f"system {sid!r}: r_max={r_max} is below stage-1 runs + floor for some cell")
        if b < int(n1.sum()):
            raise ValueError(f"system {sid!r}: budget {b} is below its {int(n1.sum())} stage-1 runs")
        if b < int(lo.sum()):
            raise ValueError(f"system {sid!r}: budget {b} is below stage-1 runs + floor ({int(lo.sum())})")
        if b > int(hi.sum()):
            warnings.warn(
                f"plan_second_stage: budget {b} of system {sid!r} exceeds r_max x cells = {int(hi.sum())}; every cell gets r_max",
                stacklevel=2,
            )
            b = int(hi.sum())
        p = (k + smoothing) / (n1 + 2.0 * smoothing)
        w = np.sqrt(p * (1.0 - p))
        n_tot = _neyman_integer(w, lo, hi, b, rng)
        out.append(
            pd.DataFrame(
                dict(
                    system_id=sid,
                    task_id=sub["task_id"].to_numpy(),
                    n_stage1=n1,
                    n_stage2=n_tot - n1,
                    n_total=n_tot,
                    p_smoothed=p,
                    weight=w,
                )
            )
        )
    return pd.concat(out, ignore_index=True)
