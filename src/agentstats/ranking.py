"""Ranking stability: bootstrap vs the full data, and agreement between independent reruns.

Two different questions are answered here:

* :func:`rank_stability` resamples the data and compares each replicate ranking with the
  *full-data* ranking.  The full-data ranking is itself noisy, so this agreement is biased upward
  (for identical systems the expected Kendall tau is 0.5, not 0).
* :func:`rerun_agreement` / :func:`rerun_agreement_model` answer "how often would an
  independent rerun change the ranking?": two independent evaluations are compared with each
  other, so identical systems give tau = 0 and P(top-1 identical) = 1/n_systems.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._utils import as_rng, check_estimand, check_int, z_two_sided
from .bootstrap import boot_system_means, check_resamplable, check_scheme
from .data import Tensor, build_tensor, prep
from .variance import VarianceComponents

_QUANTILES = (0.025, 0.25, 0.5, 0.75, 0.975)
_CHUNK = 3_000_000


def _ordinal_ranks(means: np.ndarray) -> np.ndarray:
    """0-based rank (0 = best, i.e. highest mean) per row; ties broken by column order."""
    order = np.argsort(-means, axis=-1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(means.shape[-1]), order.shape), -1)
    return ranks


def _random_tie_ranks(means: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """0-based ranks (0 = best); exact ties are broken uniformly at random (no tie bias)."""
    order = np.lexsort((rng.random(means.shape), -means), axis=-1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(means.shape[-1]), order.shape), -1)
    return ranks


def _pair_disagreement(ranks: np.ndarray, ref_ranks: np.ndarray):
    """Boolean (..., n_pairs) array: pair order differs from reference; plus pair indices."""
    iu, ju = np.triu_indices(ranks.shape[-1], k=1)
    cur = ranks[..., iu] < ranks[..., ju]
    ref = ref_ranks[..., iu] < ref_ranks[..., ju]
    return cur != ref, iu, ju


def _tau_from_ranks(ranks: np.ndarray, ref_ranks: np.ndarray) -> np.ndarray:
    dis, _, _ = _pair_disagreement(ranks, ref_ranks)
    return 1.0 - 2.0 * dis.mean(-1)


@dataclass(frozen=True)
class RankStability:
    """Result of :func:`rank_stability`.  All frames are ordered best-to-worst by full-data mean.

    ``tau`` is the per-resample Kendall tau (a strict ordering, ties broken by system order)
    against the full-data ranking; ``rank_probs`` is the P(rank = r) matrix (rows systems,
    columns ranks 1..n); ``pairwise_flip`` holds P(order of i and j differs from the full-data
    order) (diagonal 0).  Because the reference is the noisy full-data ranking, ``tau_mean`` is
    biased upward; use :func:`rerun_agreement` for rerun agreement.
    """

    systems: list
    means: pd.Series
    tau: np.ndarray
    tau_mean: float
    rank_probs: pd.DataFrame
    top_k: int
    topk_change_prob: float
    pairwise_flip: pd.DataFrame
    n_boot: int
    estimand: str


def _complete_tensor(df: pd.DataFrame) -> Tensor:
    df = prep(df)
    t = build_tensor(df)
    if t.x.shape[0] < 2:
        raise ValueError("need at least 2 systems")
    if not t.complete:
        raise ValueError(
            "every system must be evaluated on every task; restrict to common tasks (balance())"
        )
    return t


def rank_stability(
    df: pd.DataFrame,
    n_boot: int = 2000,
    estimand: str = "task_population",
    top_k: int = 1,
    rng=None,
    scheme: str = "cluster",
) -> RankStability:
    """Bootstrap the system ranking by mean score and compare with the full-data ranking.

    Each resample draws tasks (shared by all systems, so paired; ``task_population`` only),
    recomputes each system's mean and re-ranks.  Estimand ``task_population``: tasks resampled
    with replacement; ``scheme="cluster"`` (default) keeps each task's runs, ``"two_level"`` also
    resamples runs within tasks (conservative).  Estimand ``fixed_tasks``: runs resampled within
    tasks (adjusted residual resampling, see :func:`agentstats.bootstrap_ci`).  Every system must
    be evaluated on every task (runs per cell may differ).

    Bias warning: the reference is the *full-data* ranking, which is itself noisy, so the
    agreement is biased upward.  For systems with identical true means the mean tau is about 0.5
    (replicate means are the observed means plus noise of the same size as the observed
    differences), not 0.  To answer "how often would an independent rerun change the ranking"
    use :func:`rerun_agreement`.
    """
    n_boot = check_int(n_boot, "n_boot", 10)
    check_estimand(estimand)
    check_scheme(scheme)
    rng = as_rng(rng)
    t = _complete_tensor(df)
    P = t.x.shape[0]
    top_k = check_int(top_k, "top_k")
    if top_k >= P:
        raise ValueError(f"top_k must be < number of systems ({P})")
    check_resamplable(t, estimand)
    m0 = t.cell_means().mean(-1)
    reps = boot_system_means(t, n_boot, estimand, rng, scheme)
    r0 = _ordinal_ranks(m0)
    rb = _ordinal_ranks(reps)
    dis, iu, ju = _pair_disagreement(rb, r0)
    tau = 1.0 - 2.0 * dis.mean(1)
    flip = np.zeros((P, P))
    fm = dis.mean(0)
    flip[iu, ju] = fm
    flip[ju, iu] = fm
    probs = np.stack([(rb == r).mean(0) for r in range(P)], axis=1)
    topk_change = float(((rb < top_k) != (r0 < top_k)).any(1).mean())
    best_first = np.argsort(r0)
    names = [t.systems[i] for i in best_first]
    return RankStability(
        systems=names,
        means=pd.Series(m0[best_first], index=names, name="mean"),
        tau=tau,
        tau_mean=float(tau.mean()),
        rank_probs=pd.DataFrame(
            probs[best_first], index=names, columns=np.arange(1, P + 1), dtype=float
        ),
        top_k=top_k,
        topk_change_prob=topk_change,
        pairwise_flip=pd.DataFrame(flip[np.ix_(best_first, best_first)], index=names, columns=names),
        n_boot=n_boot,
        estimand=estimand,
    )


# ---------------------------------------------------------------------------------------------
# Agreement between two independent evaluations
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RerunAgreement:
    """Agreement between the system rankings of two independent evaluations.

    ``tau`` holds the per-replicate Kendall tau between the two rankings (exact ties broken at
    random); ``tau_quantiles`` its 2.5/25/50/75/97.5% quantiles; ``p_topk_same[k]`` the
    probability that the top-``k`` *sets* are identical (only ``k`` < number of systems);
    ``pairwise_flip[i, j]`` the probability that the two evaluations order systems ``i`` and
    ``j`` differently.  Frames are ordered best-to-worst by full-data mean (empirical version) or
    by index (variance-component model).  ``n_cells_skipped`` counts (system, task) cells with
    too few runs for two disjoint halves; ``n_tasks_used`` / ``n_tasks_dropped`` the tasks kept
    or removed because of them (empirical version only).
    """

    systems: list
    n_runs: int
    n_rep: int
    tau: np.ndarray
    tau_mean: float
    tau_quantiles: pd.Series
    p_top1_same: float
    p_topk_same: dict
    pairwise_flip: pd.DataFrame
    n_cells_skipped: int = 0
    n_tasks_used: int = 0
    n_tasks_dropped: int = 0
    info: dict = field(default_factory=dict)


def _agreement(mA: np.ndarray, mB: np.ndarray, rng, ks) -> dict:
    """Agreement statistics between two (n_rep, P) arrays of system means."""
    P = mA.shape[1]
    rA = _random_tie_ranks(mA, rng)
    rB = _random_tie_ranks(mB, rng)
    dis, iu, ju = _pair_disagreement(rA, rB)
    tau = 1.0 - 2.0 * dis.mean(1)
    flip = np.zeros((P, P))
    fm = dis.mean(0)
    flip[iu, ju] = fm
    flip[ju, iu] = fm
    topk = {int(k): float(((rA < k) == (rB < k)).all(1).mean()) for k in ks if k < P}
    return {"tau": tau, "flip": flip, "topk": topk}


def _check_ks(ks) -> tuple:
    return tuple(check_int(k, "k") for k in ks)


def _pack(agr, order, names, n_runs, n_rep, **kw) -> RerunAgreement:
    tau = agr["tau"]
    flip = agr["flip"][np.ix_(order, order)]
    topk = agr["topk"]
    return RerunAgreement(
        systems=list(names),
        n_runs=n_runs,
        n_rep=n_rep,
        tau=tau,
        tau_mean=float(tau.mean()),
        tau_quantiles=pd.Series(np.quantile(tau, _QUANTILES), index=_QUANTILES, name="tau"),
        p_top1_same=float(topk.get(1, 1.0)),
        p_topk_same=topk,
        pairwise_flip=pd.DataFrame(flip, index=list(names), columns=list(names)),
        **kw,
    )


def _split_halves(x, counts, n_runs, rng, reps=1):
    """Two disjoint random sets of ``n_runs`` runs per cell -> cell means ``(b, P, T)`` each.

    ``x`` is ``(P, T, R)`` (or ``(b, P, T, R)``) padded with zeros; ``counts`` is ``(P, T)``
    with every cell having ``>= 2 * n_runs`` runs.
    """
    R = x.shape[-1]
    shape = x.shape if x.ndim == 4 else (reps,) + x.shape
    keys = rng.random(shape)
    invalid = np.arange(R) >= counts[..., None]
    keys[..., invalid] = np.inf
    perm = np.argsort(keys, axis=-1)[..., : 2 * n_runs]
    sh = np.take_along_axis(np.broadcast_to(x, shape), perm, axis=-1)
    return sh[..., :n_runs].mean(-1), sh[..., n_runs:].mean(-1)


def rerun_agreement(
    df: pd.DataFrame,
    n_runs: int,
    n_rep: int = 500,
    rng=None,
    ks=(1, 3, 5),
) -> RerunAgreement:
    """How often would an independent rerun change the ranking?  (empirical split-half version)

    Estimand: ``fixed_tasks`` - the same tasks are rerun, only the runs are fresh.  For each of
    ``n_rep`` replicates and each (system, task) cell, two DISJOINT random sets of ``n_runs``
    runs are drawn; systems are ranked on each half (mean over tasks of the cell mean) and the
    two rankings are compared: Kendall tau, P(top-1 identical), P(top-``k`` sets identical) for
    ``k`` in ``ks`` (those below the number of systems) and the pairwise order-flip matrix.
    Exact ties are broken at random.

    Cells with fewer than ``2 * n_runs`` runs are skipped.  To keep systems comparable, a task is
    used only if every system's cell for it qualifies; ``n_cells_skipped``, ``n_tasks_used`` and
    ``n_tasks_dropped`` are reported (``ValueError`` if no task qualifies).  Each half has
    ``n_runs`` runs per cell, so this is the agreement of two reruns with ``n_runs`` runs; use
    :func:`rerun_agreement_model` for ``n_runs`` above half the available runs.
    """
    n_runs = check_int(n_runs, "n_runs")
    n_rep = check_int(n_rep, "n_rep")
    ks = _check_ks(ks)
    rng = as_rng(rng)
    t = _complete_tensor(df)
    ok = t.counts >= 2 * n_runs
    skipped = int((~ok).sum())
    task_ok = ok.all(0)
    if not task_ok.any():
        raise ValueError(
            f"no task has >= {2 * n_runs} runs for every system (need two disjoint halves of "
            f"{n_runs}); lower n_runs or use rerun_agreement_model"
        )
    x, counts = t.x[:, task_ok], t.counts[:, task_ok]
    P, T, R = x.shape
    chunk = max(1, _CHUNK // (P * T * R))
    mA, mB = [], []
    done = 0
    while done < n_rep:
        b = min(chunk, n_rep - done)
        a, c = _split_halves(x, counts, n_runs, rng, reps=b)
        mA.append(a.mean(-1))
        mB.append(c.mean(-1))
        done += b
    agr = _agreement(np.concatenate(mA), np.concatenate(mB), rng, ks)
    full = (x.sum(-1) / counts).mean(-1)
    order = np.argsort(-full, kind="stable")
    return _pack(
        agr, order, [t.systems[i] for i in order], n_runs, n_rep,
        n_cells_skipped=skipped, n_tasks_used=int(task_ok.sum()),
        n_tasks_dropped=int((~task_ok).sum()),
    )


def _fit_cell_probabilities(df: pd.DataFrame):
    """Empirical-Bayes cell success probabilities for binary scores (see rerun_agreement_model)."""
    df = prep(df)
    if not np.isin(df["score"].to_numpy(), (0.0, 1.0)).all():
        raise ValueError("the data-driven rerun model needs binary 0/1 scores")
    t = _complete_tensor(df)
    if t.counts.min() < 2:
        raise ValueError("every cell needs >= 2 runs to estimate the shrinkage")
    P, T, _ = t.x.shape
    if T < 2:
        raise ValueError("need at least 2 tasks")
    pi, lams = _shrink_cell_means(t.cell_means(), t.counts)
    return t, pi, lams


def _shrink_cell_means(m: np.ndarray, counts: np.ndarray):
    """Empirical-Bayes cell probabilities from a ``(P, T)`` array of cell means and run counts."""
    P, T = m.shape
    g = m.mean()
    a = m.mean(1) - g  # raw system effects
    b = m.mean(0) - g  # raw task effects
    r = m - (g + a[:, None] + b[None, :])  # raw interaction + run noise
    noise = (m * (1 - m) / (counts - 1)).mean()  # E[Var(cell mean | pi)] = pi(1-pi)/R_c

    def reliability(observed_var, noise_var):
        return float(np.sqrt(np.clip(1.0 - noise_var / observed_var, 0.0, 1.0))) if observed_var > 0 else 0.0

    lam_c = reliability((r**2).sum() / ((P - 1) * (T - 1)), noise)
    lam_p = reliability((a**2).sum() / (P - 1), noise / T)
    lam_t = reliability((b**2).sum() / (T - 1), noise / P)
    pi = np.clip(g + lam_p * a[:, None] + lam_t * b[None, :] + lam_c * r, 0.0, 1.0)
    return pi, {"system": lam_p, "task": lam_t, "interaction": lam_c}


def _model_means(pi: np.ndarray, n_runs: int, n_tasks, n_rep: int, rng):
    """Two independent ``n_runs``-run evaluations of the fitted benchmark: ``(n_rep, P)`` means each."""
    P, T = pi.shape
    n_t = T if n_tasks is None else n_tasks
    chunk = max(1, _CHUNK // (P * n_t))
    mA, mB = [], []
    done = 0
    while done < n_rep:
        b = min(chunk, n_rep - done)
        if n_tasks is None:
            p = np.broadcast_to(pi[:, None, :], (P, b, T))
        else:
            if n_t <= T:
                idx = np.argsort(rng.random((b, T)), axis=1)[:, :n_t]
            else:
                idx = rng.integers(0, T, size=(b, n_t))
            p = pi[:, idx]
        mA.append(rng.binomial(n_runs, p).mean(-1).T / n_runs)
        mB.append(rng.binomial(n_runs, p).mean(-1).T / n_runs)
        done += b
    return np.concatenate(mA), np.concatenate(mB)


NULL_TAU_TOL = 0.05  # |model tau on identical systems| above this marks the model uncalibrated
_NULL_SIMS, _NULL_REPS = 10, 200


def _null_tau(t: Tensor, n_runs: int, n_tasks, rng) -> float:
    """Model tau at ``n_runs`` when all systems are identical (should be 0 for a calibrated model).

    Every system gets the cell probabilities of the median-score system, data are simulated with
    the observed run counts, the model is refitted and its tau is averaged over a few
    simulations.  Same construction as the paper's null calibration (results/c2/model_gate.json).
    """
    cm = t.cell_means()
    base = int(np.argmin(np.abs(cm.mean(1) - np.median(cm.mean(1)))))
    p_cell = cm[base]
    taus = []
    for _ in range(_NULL_SIMS):
        m = rng.binomial(t.counts, p_cell[None, :]) / t.counts
        pi, _ = _shrink_cell_means(m, t.counts)
        mA, mB = _model_means(pi, n_runs, n_tasks, _NULL_REPS, rng)
        taus.append(_agreement(mA, mB, rng, (1,))["tau"].mean())
    return float(np.mean(taus))


def rerun_agreement_model(
    vc_or_df,
    n_runs: int,
    n_tasks=None,
    n_rep: int = 500,
    rng=None,
    ks=(1, 3, 5),
    check_null: bool = True,
) -> RerunAgreement:
    """Parametric rerun agreement, usable for any ``n_runs`` (also above half the available runs).

    .. warning::
       The data-frame model is **not calibrated under the null beyond small k**.  When the
       systems are in truth identical the fitted model still returns a positive Kendall tau
       that grows with ``n_runs`` (see ``results/c2/model_gate.json`` of the accompanying
       paper: the largest ``k`` with ``|null tau| < 0.05`` ranged from 1 to 5 across releases
       under the library's shrinkage rule).  Treat values beyond the empirical range
       (``n_runs > floor(R/2)``) as extrapolation.  ``info["calibrated"]`` records a null check
       run on the supplied data (identical systems, same run counts, same ``n_runs``; tolerance
       ``|tau| <= 0.05``), with the null tau in ``info["null_tau"]``; a ``UserWarning`` is raised
       when it fails.  Pass ``check_null=False`` to skip the check (``calibrated`` is then
       ``None``).

    Estimand: ``fixed_tasks`` for the data-frame input (the fitted benchmark is rerun); for the
    variance-component input the systems and the interaction are drawn from the fitted
    population, so the result is averaged over systems of that population.

    * ``VarianceComponents`` input: Gaussian random-effects model.  Per replicate draw system
      effects ``a_p ~ N(0, s_p)`` and interactions ``c_pt ~ N(0, s_pt)`` (shared by both
      evaluations: same tasks); each evaluation adds run noise ``N(0, s_e / (n_runs * n_t))`` to
      every system mean.  The task effect cancels in rankings.  ``n_tasks`` defaults to
      ``vc.n_t``; the number of systems is ``vc.n_p``.
    * Data-frame input (binary 0/1 scores): fit cell success probabilities ``pi_pt`` and simulate
      two independent evaluations ``Binomial(n_runs, pi_pt) / n_runs``.  Raw cell means are
      noisy; using them as probabilities would treat noise as real differences between systems
      (and shared system x task structure) and overstate agreement - for identical systems the
      model would report tau > 0.  Every effect of the additive decomposition
      ``m_pt = m.. + a_p + b_t + r_pt`` is therefore shrunk by empirical Bayes (method of
      moments), ``effect * max(0, 1 - noise_var / observed_var)``, where ``noise_var`` is the
      binomial noise of that effect's estimate, ``N = mean m(1-m)/(R_c - 1)`` for ``r_pt``,
      ``N / n_t`` for ``a_p`` and ``N / n_p`` for ``b_t`` (a fixed-benchmark rerun: the
      interaction is part of the truth, only the run noise is noise).  The shrinkage factors are
      returned in ``info["shrinkage"]``.  ``n_tasks`` (optional) simulates a smaller/larger
      benchmark by drawing that many tasks per replicate (without replacement if <= the
      available tasks, otherwise with replacement), shared by both evaluations.

    Output as :func:`rerun_agreement`; ``info`` holds the fitted shrinkage factors and the
    ``calibrated`` flag (data-frame input).  For ``VarianceComponents`` input ``calibrated`` is
    ``True``: the Gaussian model has no shrinkage step to bias the null, and its null (``s_p = 0``)
    gives tau = 0 by construction.
    """
    n_runs = check_int(n_runs, "n_runs")
    n_rep = check_int(n_rep, "n_rep")
    ks = _check_ks(ks)
    rng = as_rng(rng)
    if isinstance(vc_or_df, VarianceComponents):
        vc = vc_or_df
        P = vc.n_p
        T = vc.n_t if n_tasks is None else check_int(n_tasks, "n_tasks")
        a = rng.normal(0.0, np.sqrt(vc.sigma2_p), (n_rep, P))
        c = rng.normal(0.0, np.sqrt(vc.sigma2_pt), (n_rep, P, T)).mean(-1)
        sd = np.sqrt(vc.sigma2_e / (n_runs * T))
        mA = a + c + rng.normal(0.0, sd, (n_rep, P))
        mB = a + c + rng.normal(0.0, sd, (n_rep, P))
        names = [f"system_{i}" for i in range(P)]
        agr = _agreement(mA, mB, rng, ks)
        return _pack(agr, np.arange(P), names, n_runs, n_rep, n_tasks_used=T,
                     info={"model": "variance_components", "calibrated": True})
    t, pi, lams = _fit_cell_probabilities(vc_or_df)
    P, T = pi.shape
    n_t = T if n_tasks is None else check_int(n_tasks, "n_tasks", 2)
    mA, mB = _model_means(pi, n_runs, None if n_tasks is None else n_t, n_rep, rng)
    calibrated, null_tau = None, None
    if check_null:
        null_tau = _null_tau(t, n_runs, None if n_tasks is None else n_t, np.random.default_rng(0))
        calibrated = bool(abs(null_tau) <= NULL_TAU_TOL)
        if not calibrated:
            warnings.warn(
                f"rerun_agreement_model: at n_runs={n_runs} the model gives tau={null_tau:.3f} for "
                f"identical systems (tolerance {NULL_TAU_TOL}); it is not calibrated under the null "
                "here, so its agreement is inflated. See the docstring.",
                stacklevel=2,
            )
    agr = _agreement(mA, mB, rng, ks)
    full = t.cell_means().mean(-1)
    order = np.argsort(-full, kind="stable")
    return _pack(agr, order, [t.systems[i] for i in order], n_runs, n_rep,
                 n_tasks_used=n_t, info={"model": "empirical_bayes_binomial", "shrinkage": lams,
                       "calibrated": calibrated, "null_tau": null_tau, "null_tolerance": NULL_TAU_TOL})


def subsample_curve(
    df: pd.DataFrame,
    n_runs_grid,
    n_tasks_grid,
    n_rep: int = 200,
    rng=None,
    vs_full: bool = False,
) -> pd.DataFrame:
    """How ranking quality degrades with fewer runs and tasks (tidy frame for plotting).

    For every ``(n_r, n_t)`` pair, ``n_rep`` times: draw ``n_t`` tasks without replacement, then
    in every cell two DISJOINT sets of ``n_r`` runs (as in :func:`rerun_agreement`; the same
    tasks are used for both halves, estimand ``fixed_tasks``).  Columns: ``n_runs, n_tasks,
    n_rep, reference, mean_tau`` (Kendall tau between the two half-rankings; 0 for identical
    systems), ``mean_halfwidth`` (mean over systems, replicates and halves of the
    normal-approximation 95% half-width ``1.96 * sd(task means) / sqrt(n_t)``, the
    ``task_population`` estimand), ``top1_flip_rate`` (fraction of replicates whose top system
    differs between the halves; ties broken at random).  Requires a balanced, complete design
    with ``2 * max(n_runs_grid) <= R``.

    With ``vs_full=True`` one subsample (``n_r`` runs, no disjoint half needed, ``n_r <= R``) is
    compared with the full-data ranking instead (tau and flip rate vs the full data;
    ``reference="vs_full"``).  That metric is biased upward: the subsample is part of the
    reference, and the reference ranking is itself noisy (it reaches tau = 1 at the full size by
    construction), so it must not be read as rerun agreement.
    """
    n_rep = check_int(n_rep, "n_rep")
    rng = as_rng(rng)
    t = _complete_tensor(df)
    if not t.balanced:
        raise ValueError("subsample_curve needs a balanced design; call balance() first")
    P, T, R = t.x.shape
    runs = sorted({check_int(v, "n_runs_grid value") for v in np.atleast_1d(n_runs_grid)})
    tasks = sorted({check_int(v, "n_tasks_grid value", 2) for v in np.atleast_1d(n_tasks_grid)})
    need = runs[-1] if vs_full else 2 * runs[-1]
    if need > R:
        raise ValueError(
            f"n_runs_grid value {runs[-1]} exceeds available runs per cell ({R})"
            + ("" if vs_full else ": disjoint halves need 2 * n_runs <= R (or use vs_full=True)")
        )
    if tasks[-1] > T:
        raise ValueError(f"n_tasks_grid value {tasks[-1]} exceeds available tasks ({T})")
    z = z_two_sided(0.95)
    m0 = t.x.mean(-1).mean(-1)
    r0 = _ordinal_ranks(m0)
    top0 = int(np.argmin(r0))
    counts = t.counts
    rows = []
    for n_r in runs:
        for n_t in tasks:
            chunk = max(1, 4_000_000 // (P * n_t * R))
            taus, hws, flips = [], [], []
            done = 0
            while done < n_rep:
                c = min(chunk, n_rep - done)
                tsel = np.argsort(rng.random((c, T)), axis=1)[:, :n_t]  # (c, n_t)
                xs = t.x[:, tsel].transpose(1, 0, 2, 3)  # (c, P, n_t, R)
                if vs_full:
                    rsel = np.argsort(rng.random(xs.shape), axis=-1)[..., :n_r]
                    cell = np.take_along_axis(xs, rsel, axis=-1).mean(-1)  # (c, P, n_t)
                    rk = _random_tie_ranks(cell.mean(-1), rng)
                    taus.append(_tau_from_ranks(rk, r0))
                    flips.append(np.argmin(rk, axis=1) != top0)
                    hws.append(z * cell.std(-1, ddof=1).mean(-1) / np.sqrt(n_t))
                else:
                    cA, cB = _split_halves(xs, np.broadcast_to(counts[:, :1], (P, n_t)), n_r, rng)
                    agr = _agreement(cA.mean(-1), cB.mean(-1), rng, (1,))
                    taus.append(agr["tau"])
                    flips.append(_flip_indicator(cA.mean(-1), cB.mean(-1), rng))
                    hws.append(z * (cA.std(-1, ddof=1) + cB.std(-1, ddof=1)).mean(-1) / (2 * np.sqrt(n_t)))
                done += c
            rows.append(
                {
                    "n_runs": n_r,
                    "n_tasks": n_t,
                    "n_rep": n_rep,
                    "reference": "vs_full" if vs_full else "half_vs_half",
                    "mean_tau": float(np.concatenate(taus).mean()),
                    "mean_halfwidth": float(np.concatenate(hws).mean()),
                    "top1_flip_rate": float(np.concatenate(flips).mean()),
                }
            )
    return pd.DataFrame(rows)


def _flip_indicator(mA: np.ndarray, mB: np.ndarray, rng) -> np.ndarray:
    """Per-replicate indicator that the top system differs between two evaluations."""
    return _random_tie_ranks(mA, rng).argmin(1) != _random_tie_ranks(mB, rng).argmin(1)
