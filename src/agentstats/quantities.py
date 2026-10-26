"""The paper's named quantities as first-class functions.

* Q1 :func:`run_disagreement`: run disagreement rate ``D`` and run share ``rho_run``.
* Q3 :func:`reversal_probability`, :func:`pair_z`, :func:`pair_z_table`: probability that the
  order of two systems flips on a rerun.
* Q4 :func:`runs_needed`, :func:`fit_se_curve`: runs needed to separate a gap, and the
  run-variance curve ``s_e(p) = kappa p (1 - p)`` it is evaluated on.  The design MDE is
  :func:`agentstats.mde`; fixed-task reliability is the ``fixed_tasks`` output of
  :func:`agentstats.d_study`.

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

from ._utils import check_estimand, check_int, check_level
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

        s_pt = max(MS_pt* - s_e * mean_c(1 / R_c), 0)

    where ``MS_pt*`` is the interaction mean square of the two-way table of cell means, with
    ``(P - 1)(T - 1)`` degrees of freedom.  Since ``E[MS_pt*] = s_pt + s_e * mean_c(1 / R_c)``
    exactly, this is unbiased before truncation and equals the ANOVA estimator of
    :func:`variance_components` on balanced data.  On an incomplete design ``s_pt`` and
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
        spt = max(float(ms_pt - s_e * (1.0 / t.counts).mean()), 0.0)
        method = "unweighted cell-mean interaction MS minus s_e * mean(1/R_c), truncated at 0"
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
            v = v[np.isfinite(v)]
            if v.size == 0:
                raise ValueError(
                    f"system {sid!r} has no shared task with >= 2 runs to estimate s_e; pass s_e"
                )
            se_i = float(v.mean())
        elif isinstance(s_e, Mapping):
            if sid not in s_e:
                raise ValueError(f"s_e has no entry for system {sid!r}")
            se_i = float(s_e[sid])
        else:
            se_i = float(s_e)
        if not (np.isfinite(se_i) and se_i >= 0):
            raise ValueError("s_e must be finite and >= 0")
        parts.append(se_i * float((1.0 / t.counts[i][both]).mean()))
    se_diff = float(np.sqrt(sum(parts) / T))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = gap / se_diff if se_diff > 0 else (0.0 if gap == 0 else float("inf") * np.sign(gap))
    return PairZ(t.systems[ia], t.systems[ib], gap, se_diff, float(z), reversal_probability(z), T)


def pair_z(df: pd.DataFrame, sys_a: str, sys_b: str, s_e: Mapping | float | None = None) -> PairZ:
    """Pair-level ``z`` of the gap between two systems, with the fixed-task standard error.

    Estimand: ``fixed_tasks``.  Over the ``T`` tasks both systems share,
    ``gap = mean_t(m_a,t - m_b,t)`` (cell means) and

        SE_diff^2 = ( s_e,a * mean_t(1 / R_a,t) + s_e,b * mean_t(1 / R_b,t) ) / T

    which is ``(s_e,a + s_e,b) / (T k)`` when every cell has ``k`` runs.  ``s_e,a`` is the mean
    unbiased run variance of the system over its shared-task cells with >= 2 runs, unless ``s_e``
    is given (a float for both systems, or a mapping ``system_id -> s_e``), which is required
    when a system has no cell with 2 runs, for example a single-run evaluation using the
    full-data ``s_e``.  ``z = gap / SE_diff`` is signed; ``psi = reversal_probability(z)``.
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
