"""Named quantities (Q1, Q3, Q4) and cross-checks against the paper-side implementation.

The cross-checks read the prepared real data in ``results/prep`` (built from ``data/processed``)
and the paper code in ``analysis/y_stats_ext.py``; they skip when either is absent (for example
when the package is tested from an sdist outside the research repository).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import agentstats as ast
from _sim import df_from_array

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / "results" / "prep"
ANALYSIS = ROOT / "analysis"

have_paper = (ANALYSIS / "y_stats_ext.py").exists() and (PREP / "summary.json").exists()
needs_paper = pytest.mark.skipif(not have_paper, reason="paper repository data/analysis not present")

MULTI = ["swev_rep", "tb2", "deepswe", "tau2_airline", "tau2_retail", "tau2_telecom", "tau2_banking"]
PER_SYSTEM = ["gaia", "frames"]


@pytest.fixture(scope="module")
def ext():
    if not have_paper:
        pytest.skip("paper analysis not present")
    sys.path.insert(0, str(ANALYSIS))
    try:
        import y_stats_ext
    finally:
        sys.path.remove(str(ANALYSIS))
    return y_stats_ext


@pytest.fixture(scope="module")
def paper():
    """Paper-side modules (y_c1_variance, y_c3_dstudy); only pure helper functions are used."""
    if not have_paper:
        pytest.skip("paper analysis not present")
    sys.path.insert(0, str(ANALYSIS))
    try:
        import y_c1_variance
        import y_c3_dstudy
    finally:
        sys.path.remove(str(ANALYSIS))
    return y_c1_variance, y_c3_dstudy


def _load(key: str, balanced: bool) -> pd.DataFrame:
    p = PREP / (f"{key}.parquet" if balanced else f"{key}__unbalanced.parquet")
    if not p.exists():
        pytest.skip(f"{p.name} not present")
    return pd.read_parquet(p)


# ------------------------------------------------------------------------------ Q1: run disagreement


def _gauss_ragged(rng, P, T, s_p, s_t, s_pt, s_e, r_lo=1, r_hi=6):
    a = rng.normal(0, np.sqrt(s_p), P)
    b = rng.normal(0, np.sqrt(s_t), T)
    c = rng.normal(0, np.sqrt(s_pt), (P, T))
    R = rng.integers(r_lo, r_hi + 1, (P, T))
    rows = []
    for p in range(P):
        for t in range(T):
            mu = 0.5 + a[p] + b[t] + c[p, t]
            for r in range(R[p, t]):
                rows.append((f"s{p:02d}", f"t{t:04d}", r, mu + rng.normal(0, np.sqrt(s_e))))
    d = pd.DataFrame(rows, columns=["system_id", "task_id", "run_id", "score"])
    d.insert(0, "benchmark", "b")
    d["run_id"] = d["run_id"].astype("int64")
    return d


def test_D_cell_formula_on_binary_by_hand():
    # cell A: R=4, k=1 -> 2*1*3/(4*3) = 0.5 ; cell B: R=3, k=3 -> 0 ; cell C: R=1 skipped
    rows = [("s", "A", r, float(r == 0)) for r in range(4)]
    rows += [("s", "B", r, 1.0) for r in range(3)] + [("s", "C", 0, 1.0)]
    df = pd.DataFrame(rows, columns=["system_id", "task_id", "run_id", "score"])
    df.insert(0, "benchmark", "b")
    with pytest.warns(UserWarning, match="incomplete|< 2 systems"):
        rd = ast.run_disagreement(df)
    assert rd.D == pytest.approx((0.5 + 0.0) / 2)
    assert rd.s_e == pytest.approx(rd.D / 2)
    assert rd.n_cells == 2
    assert np.isnan(rd.rho_run) and "unavailable" in rd.s_pt_method
    assert rd.by_system.loc[0, "D"] == pytest.approx(rd.D)
    with pytest.raises(ValueError, match=">= 2 runs"):
        ast.run_disagreement(df[df.task_id == "C"])


def test_run_disagreement_balanced_matches_variance_components_and_pairs():
    rng = np.random.default_rng(0)
    y = (rng.random((6, 80, 5)) < rng.beta(1.5, 1.5, (6, 80, 1))).astype(float)
    df = df_from_array(y)
    rd = ast.run_disagreement(df)
    vc = ast.variance_components(df)
    assert rd.s_e == pytest.approx(vc.sigma2_e, abs=1e-12)
    assert rd.s_pt == pytest.approx(vc.sigma2_pt, abs=1e-12)
    assert rd.rho_run == pytest.approx(vc.sigma2_e / (vc.sigma2_pt + vc.sigma2_e), abs=1e-12)
    k = y.sum(-1)
    assert rd.D == pytest.approx((2 * k * (5 - k) / (5 * 4)).mean(), abs=1e-12)
    assert list(rd.by_system.columns) == ["system_id", "D", "s_e", "mean", "n_tasks", "n_cells_used"]
    assert rd.by_system.D.mean() == pytest.approx(rd.D)  # equal cells per system
    # an explicit s_pt overrides the estimate
    r2 = ast.run_disagreement(df, s_pt=0.0)
    assert r2.rho_run == 1.0 and r2.s_pt_method == "supplied"
    with pytest.raises(ValueError):
        ast.run_disagreement(df, s_pt=-1.0)


def test_s_pt_estimator_unbiased_for_ragged_cells():
    rng = np.random.default_rng(7)
    s_pt_true, s_e_true = 0.002, 0.004
    est, ese = [], []
    for _ in range(60):
        d = _gauss_ragged(rng, 8, 100, 0.001, 0.003, s_pt_true, s_e_true)
        rd = ast.run_disagreement(d)
        est.append(rd.s_pt)
        ese.append(rd.s_e)
    assert np.mean(ese) == pytest.approx(s_e_true, rel=0.03)
    assert np.mean(est) == pytest.approx(s_pt_true, abs=1e-4)


@needs_paper
@pytest.mark.parametrize("key", MULTI + PER_SYSTEM)
def test_run_disagreement_matches_paper_code_on_real_data(ext, key):
    df = _load(key, balanced=False)
    cells = ext.cell_table(df)
    se_rag, n_used = ext.ragged_se(cells)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd = ast.run_disagreement(df)
    assert rd.s_e == pytest.approx(se_rag, rel=1e-12)
    assert rd.D == pytest.approx(2 * se_rag, rel=1e-12)
    assert rd.n_cells == n_used
    tbl = ext.se_by_system(cells).set_index("system_id")
    got = rd.by_system.set_index("system_id").loc[tbl.index]
    np.testing.assert_allclose(got["s_e"], tbl["s_e"], rtol=1e-12)
    np.testing.assert_allclose(got["D"], 2 * tbl["s_e"], rtol=1e-12)
    np.testing.assert_allclose(got["mean"], tbl["mean"], rtol=1e-12)
    np.testing.assert_array_equal(got["n_cells_used"], tbl["n_cells_used"])
    # exact pair formula D_c = 2k(R-k)/(R(R-1)) on binary scores
    if np.isin(df.score.to_numpy(), (0.0, 1.0)).all():
        g = df.groupby(["system_id", "task_id"]).score.agg(["sum", "count"])
        g = g[g["count"] >= 2]
        dc = 2 * g["sum"] * (g["count"] - g["sum"]) / (g["count"] * (g["count"] - 1))
        assert rd.D == pytest.approx(float(dc.mean()), rel=1e-12)
    # s_e(p) curve: same numbers as the analysis fit (weights = cells used)
    fc = ext.fit_se_curve(tbl["mean"].to_numpy(), tbl["s_e"].to_numpy(), tbl["n_cells_used"].to_numpy())
    curve = rd.se_curve()
    for name in ("kappa", "r2"):
        assert getattr(curve, name) == pytest.approx(fc[name], rel=1e-9)
    assert curve.kappa_se == pytest.approx(fc["kappa_se"], rel=1e-9)
    assert curve.n_systems == fc["n_systems"]


@needs_paper
@pytest.mark.parametrize("key", MULTI)
def test_rho_run_matches_paper_on_real_data(ext, paper, key):
    c1, _ = paper
    bal = _load(key, balanced=True)
    unb = _load(key, balanced=False)
    vc = ast.variance_components(bal)
    sig = dict(p=vc.sigma2_p, t=vc.sigma2_t, pt=vc.sigma2_pt, e=vc.sigma2_e)
    want = c1.derived(sig, vc.n_t)
    # balanced data: identical to the crossed-model quantity used in the paper
    rd = ast.run_disagreement(bal)
    assert rd.s_e == pytest.approx(vc.sigma2_e, rel=1e-10)
    assert rd.s_pt == pytest.approx(vc.sigma2_pt, rel=1e-8, abs=1e-12)
    assert rd.rho_run == pytest.approx(want["rho_run"], rel=1e-8)
    assert rd.D == pytest.approx(want["D_balanced"], rel=1e-10)
    # paper headline (ragged s_e, balanced s_pt): reproduced with the s_pt override
    se_rag, _ = ext.ragged_se(ext.cell_table(unb))
    rr = ast.run_disagreement(unb, s_pt=vc.sigma2_pt)
    assert rr.rho_run == pytest.approx(se_rag / (vc.sigma2_pt + se_rag), rel=1e-10)
    assert rr.D == pytest.approx(2 * se_rag, rel=1e-10)
    # the built-in ragged s_pt estimator on the unbalanced release stays close to the balanced one
    est = ast.run_disagreement(unb)
    if np.isfinite(est.rho_run):
        assert abs(est.rho_run - rr.rho_run) < 0.15


# ------------------------------------------------------------------------------ fixed-task reliability


@needs_paper
@pytest.mark.parametrize("key", ["swev_rep", "tb2", "tau2_retail"])
def test_d_study_fixed_tasks_matches_paper_code(ext, key):
    vc = ast.variance_components(_load(key, balanced=True))
    n_t, n_r = np.array([[20], [vc.n_t], [1000]]), np.array([1, 2, 5, 32])
    ds = ast.d_study(vc, n_t, n_r)
    ref = ext.d_study_fixed(vc.sigma2_p, vc.sigma2_pt, vc.sigma2_e, n_t, n_r)
    np.testing.assert_allclose(ds.fixed_tasks, ref, rtol=1e-12)
    assert (ds.fixed_tasks >= ds.e_rho2 - 1e-12).all()


def test_d_study_fixed_tasks_hand_computed():
    from agentstats.variance import VarianceComponents

    vc = VarianceComponents(0.02, 0.05, 0.01, 0.03, 0, 0, 0, 0, 8, 300, 10)
    ds = ast.d_study(vc, n_t=100, n_r=5)
    sp = 0.02 + 0.01 / 100
    assert float(ds.fixed_tasks) == pytest.approx(sp / (sp + 0.03 / 500))


# ------------------------------------------------------------------------------ Q3: reversal


def test_reversal_probability_values():
    assert ast.reversal_probability(0.0) == pytest.approx(0.5)
    assert ast.reversal_probability(-2.0) == ast.reversal_probability(2.0)
    assert ast.reversal_probability(2.0) == pytest.approx(stats.norm.cdf(-2.0 / np.sqrt(2)))
    z = np.array([0.0, 1.0, np.nan])
    out = ast.reversal_probability(z)
    assert out.shape == (3,) and np.isnan(out[2])


@needs_paper
def test_pair_z_matches_paper_split_half_on_real_data(ext):
    y_df = _load("swev_rep", balanced=True)
    t = ast.data.build_tensor(y_df)
    y = t.x  # (P, T, R)
    k, seed = 3, 5
    P, T, _ = y.shape
    A, _B = ext.halves(y, k, np.random.default_rng(seed), 1)
    z_paper, _rev, pred_paper = ext.psi_split(y, k, np.random.default_rng(seed), 1)
    tab = ast.pair_z_table(df_from_array(A[0]))
    assert len(tab) == P * (P - 1) // 2
    keep = tab["gap"].to_numpy() != 0
    np.testing.assert_allclose(np.abs(tab["z"].to_numpy())[keep], z_paper, rtol=1e-10)
    np.testing.assert_allclose(tab["psi"].to_numpy()[keep], pred_paper, rtol=1e-10)
    # single pair equals its table row
    one = ast.pair_z(df_from_array(A[0]), "s002", "s007")
    row = tab[(tab.system_a == "s002") & (tab.system_b == "s007")].iloc[0]
    assert one.z == pytest.approx(row.z) and one.se_diff == pytest.approx(row.se_diff)
    # k = 1: the full-data per-system s_e is supplied, as in the paper
    se_full = ext.se_by_system(ext.cell_table(y_df)).set_index("system_id")["s_e"]
    A1, _ = ext.halves(y, 1, np.random.default_rng(seed), 1)
    z1, _r, _p = ext.psi_split(y, 1, np.random.default_rng(seed), 1, se_full=se_full.to_numpy())
    d1 = df_from_array(A1[0])
    with pytest.raises(ValueError, match="pass s_e"):
        ast.pair_z(d1, "s000", "s001")
    tab1 = ast.pair_z_table(d1, s_e={f"s{i:03d}": se_full.iloc[i] for i in range(P)})
    keep1 = tab1["gap"].to_numpy() != 0
    np.testing.assert_allclose(np.abs(tab1["z"].to_numpy())[keep1], z1, rtol=1e-10)


def test_pair_z_ragged_formula_by_hand():
    rng = np.random.default_rng(3)
    d = _gauss_ragged(rng, 2, 40, 0.001, 0.003, 0.002, 0.004, r_lo=2, r_hi=5)
    pz = ast.pair_z(d, "s00", "s01")
    g = d.groupby(["system_id", "task_id"]).score.agg(["mean", "var", "count"])
    a, b = g.loc["s00"], g.loc["s01"]
    gap = (a["mean"] - b["mean"]).mean()
    T = len(a)
    se2 = ((a["var"] / a["count"]).sum() + (b["var"] / b["count"]).sum()) / T**2
    assert pz.gap == pytest.approx(gap) and pz.se_diff == pytest.approx(np.sqrt(se2))
    assert pz.z == pytest.approx(gap / np.sqrt(se2))
    assert pz.psi == pytest.approx(stats.norm.cdf(-abs(pz.z) / np.sqrt(2)))
    with pytest.raises(ValueError):
        ast.pair_z(d, "s00", "s00")


# ---- ragged fixed-task variance (fixed in 0.2.1): T^-2 sum_t s2_t / R_t, not mean(s2) * mean(1/R)


def _long(cells):
    """Long frame from ``{(system, task): [scores...]}``."""
    rows = [(sid, tid, r, float(v)) for (sid, tid), vs in cells.items() for r, v in enumerate(vs)]
    df = pd.DataFrame(rows, columns=["system_id", "task_id", "run_id", "score"])
    df.insert(0, "benchmark", "b")
    return df


def test_pair_z_variance_exactly_unbiased_two_task_counterexample():
    # Each system: task A ~ Bernoulli(1/2) with 2 runs, task B deterministic (all 1) with 10 runs.
    # True Var(gap) = 2 * (1/4) * Var(mean of 2 Bernoulli(1/2)) = 2 * (1/4) * (1/8) = 1/16.
    # The pre-0.2.1 formula has expectation 3/80 (40% too low); enumerate all outcomes exactly.
    ks = [(0, 1 / 4), (1, 1 / 2), (2, 1 / 4)]
    e_new = e_old = 0.0
    var_gap = 0.0
    for ka, pa in ks:
        for kb, pb in ks:
            w = pa * pb
            cells = {}
            for sid, k in (("a", ka), ("b", kb)):
                cells[(sid, "A")] = [1.0] * k + [0.0] * (2 - k)
                cells[(sid, "B")] = [1.0] * 10
            pz = ast.pair_z(_long(cells), "a", "b")
            e_new += w * pz.se_diff**2
            var_gap += w * pz.gap**2  # true mean gap is 0
            old = 0.0
            for sid, k in (("a", ka), ("b", kb)):
                s2 = [0.5 if k == 1 else 0.0, 0.0]
                old += np.mean(s2) * np.mean([1 / 2, 1 / 10]) / 2
            e_old += w * old
    assert var_gap == pytest.approx(1 / 16, abs=1e-15)
    assert e_new == pytest.approx(1 / 16, abs=1e-15)
    assert e_old == pytest.approx(3 / 80, abs=1e-15)  # documents the old bias


def _old_pair_se2(df, a, b):
    """Pre-0.2.1 pair_z formula, kept here only to prove balanced designs are unchanged."""
    g = df.groupby(["system_id", "task_id"]).score.agg(["var", "count"])
    out = 0.0
    for sid in (a, b):
        x = g.loc[sid]
        out += x["var"].mean() * (1.0 / x["count"]).mean()
    return out / len(g.loc[a])


@pytest.mark.parametrize("kind", ["binary", "continuous"])
def test_balanced_design_unchanged_by_ragged_fix(kind):
    rng = np.random.default_rng(11)
    P, T, R = 4, 50, 6
    if kind == "binary":
        y = (rng.random((P, T, R)) < rng.beta(0.6, 0.6, (P, T, 1))).astype(float)
    else:
        y = np.clip(rng.normal(0.5, 0.2, (P, T, R)), 0.0, 1.0)
    df = df_from_array(y)
    for a, b in (("s000", "s001"), ("s001", "s003")):
        assert ast.pair_z(df, a, b).se_diff**2 == pytest.approx(_old_pair_se2(df, a, b), rel=1e-12)
    tab = ast.pair_z_table(df)
    for _, row in tab.iterrows():
        assert row.se_diff**2 == pytest.approx(_old_pair_se2(df, row.system_a, row.system_b), rel=1e-12)
    # supplied s_e: old = s_e/(T R) per system, new = same
    pz = ast.pair_z(df, "s000", "s001", s_e=0.07)
    assert pz.se_diff**2 == pytest.approx(2 * 0.07 / (T * R), rel=1e-12)
    # s_pt: old correction s_e * mean(1/R) = s_e / R equals the new mean(s2/R)
    rd = ast.run_disagreement(df)
    ms_pt = _interaction_ms(y.mean(-1))
    assert rd.s_pt == pytest.approx(max(ms_pt - rd.s_e / R, 0.0), rel=1e-12)
    vc = ast.variance_components(df)
    assert rd.s_pt == pytest.approx(vc.sigma2_pt, abs=1e-12)


def _interaction_ms(cm):
    P, T = cm.shape
    r = cm - cm.mean(1, keepdims=True) - cm.mean(0, keepdims=True) + cm.mean()
    return float((r**2).sum() / ((P - 1) * (T - 1)))


def _ragged_binary_df(rng, pi, R):
    """Binary runs for fixed cell probabilities ``pi`` (P, T) with run counts ``R`` (P, T)."""
    P, T = pi.shape
    rows = []
    for p in range(P):
        for t in range(T):
            draws = (rng.random(R[p, t]) < pi[p, t]).astype(float)
            rows.extend((f"s{p}", f"t{t:03d}", r, v) for r, v in enumerate(draws))
    df = pd.DataFrame(rows, columns=["system_id", "task_id", "run_id", "score"])
    df.insert(0, "benchmark", "b")
    return df


def test_pair_z_se_coverage_heterogeneous_binary_ragged_runs():
    # Fixed tasks, binary cells: probabilities near 1/2 get 2 runs, extreme ones get 12, so cell
    # variance and run count are strongly associated (the case the old formula got wrong).
    rng = np.random.default_rng(2026)
    T = 80
    pi = np.stack([rng.beta(0.5, 0.5, T), rng.beta(0.5, 0.5, T)])
    pi = np.clip(pi, 0.02, 0.98)
    R = np.where(np.abs(pi - 0.5) < 0.3, 2, 12)
    truth = float((pi[0] - pi[1]).mean())
    var_true = float((pi * (1 - pi) / R).sum() / T**2)
    covered = covered_old = 0
    se2_new, se2_old = [], []
    reps = 1500
    for _ in range(reps):
        df = _ragged_binary_df(rng, pi, R)
        pz = ast.pair_z(df, "s0", "s1")
        covered += abs(pz.gap - truth) <= 1.959964 * pz.se_diff
        covered_old += abs(pz.gap - truth) <= 1.959964 * np.sqrt(_old_pair_se2(df, "s0", "s1"))
        se2_new.append(pz.se_diff**2)
        se2_old.append(_old_pair_se2(df, "s0", "s1"))
    # the estimator is unbiased for the true fixed-task variance; the old one is not
    assert np.mean(se2_new) == pytest.approx(var_true, rel=0.04)
    assert abs(np.mean(se2_old) / var_true - 1) > 0.10
    cov, cov_old = covered / reps, covered_old / reps
    assert 0.93 <= cov <= 0.97, cov  # nominal 0.95, MC s.e. ~ 0.006
    assert cov_old < cov


def test_s_pt_correction_unbiased_heterogeneous_binary_ragged_runs():
    rng = np.random.default_rng(5)
    P, T = 4, 60
    base = rng.beta(0.6, 0.6, (1, T))
    pi = np.clip(base + rng.normal(0, 0.25, (P, T)), 0.02, 0.98)  # strong interaction
    R = np.where(np.abs(pi - 0.5) < 0.3, 2, 9)
    target = _interaction_ms(pi)  # E[MS_pt*] - mean(sigma2/R) is the interaction MS of the true table
    est = []
    for _ in range(300):
        df = _ragged_binary_df(rng, pi, R)
        est.append(ast.run_disagreement(df).s_pt)
    assert np.mean(est) == pytest.approx(target, rel=0.05)


# ------------------------------------------------------------------------------ Q4: runs needed


def test_fit_se_curve_recovers_kappa_and_evaluates():
    rng = np.random.default_rng(1)
    m = rng.uniform(0.1, 0.9, 40)
    c = ast.fit_se_curve(m, 0.7 * m * (1 - m) + rng.normal(0, 0.002, 40))
    assert c.kappa == pytest.approx(0.7, abs=0.03) and c.n_systems == 40
    assert c(0.5) == pytest.approx(c.kappa * 0.25)
    lo, hi = c.s_e_band(0.5)
    assert lo < c(0.5) < hi
    with pytest.raises(ValueError):
        ast.fit_se_curve([0.5], [0.1, 0.2])


def test_runs_needed_closed_forms():
    zk = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    r = ast.runs_needed(0.02, 0.2, 500)                       # default criterion: power
    assert r.criterion == "power" and r.z == pytest.approx(zk)
    assert r.n_r_cont == pytest.approx(2 * 0.2 * zk**2 / (500 * 0.02**2))
    assert r.n_r == int(np.ceil(r.n_r_cont)) and r.determinable
    # the returned n_r is the smallest integer whose design MDE is <= d
    se = lambda n: np.sqrt(2 * 0.2 / (500 * n))
    assert zk * se(r.n_r) <= 0.02 + 1e-12
    if r.n_r > 1:
        assert zk * se(r.n_r - 1) > 0.02
    assert ast.mde(0.0, 0.2, 500, r.n_r, estimand="fixed_tasks") <= 0.02 + 1e-12
    # alpha / power arguments
    r90 = ast.runs_needed(0.02, 0.2, 500, alpha=0.05, power=0.9)
    assert r90.n_r_cont == pytest.approx(r.n_r_cont * ((stats.norm.ppf(0.975) + stats.norm.ppf(0.9)) / zk) ** 2)
    # reversal criterion: Phi(-d / SE) <= risk
    z = stats.norm.ppf(0.95)
    rv = ast.runs_needed(0.02, 0.2, 500, criterion="reversal")
    assert rv.criterion == "reversal" and rv.z == pytest.approx(z)
    assert rv.n_r_cont == pytest.approx(2 * 0.2 * z**2 / (500 * 0.02**2))
    assert stats.norm.cdf(-0.02 / se(rv.n_r)) <= 0.05 + 1e-12
    if rv.n_r > 1:
        assert stats.norm.cdf(-0.02 / se(rv.n_r - 1)) > 0.05
    assert rv.n_r <= r.n_r and r.n_r_cont / rv.n_r_cont == pytest.approx((zk / z) ** 2)
    # 1/d^2 scaling
    r1, r2 = ast.runs_needed(0.01, 0.2, 500), ast.runs_needed(0.02, 0.2, 500)
    assert r1.n_r_cont == pytest.approx(4 * r2.n_r_cont)
    # task population: needs enough tasks, else not determinable
    ok = ast.runs_needed(0.02, 0.2, 5000, estimand="task_population", sigma2_delta_task=0.01)
    assert ok.determinable and ok.n_r >= 1
    assert ast.mde(0.01, 0.2, 5000, ok.n_r, estimand="task_population") <= 0.02 + 1e-12
    if ok.n_r > 1:
        assert ast.mde(0.01, 0.2, 5000, ok.n_r - 1, estimand="task_population") > 0.02
    okr = ast.runs_needed(0.02, 0.2, 5000, estimand="task_population", sigma2_delta_task=0.01, criterion="reversal")
    var = (0.01 + 2 * 0.2 / okr.n_r) / 5000
    assert stats.norm.cdf(-0.02 / np.sqrt(var)) <= 0.05 + 1e-12
    no = ast.runs_needed(0.02, 0.2, 100, estimand="task_population", sigma2_delta_task=0.1)
    assert not no.determinable and no.n_r is None
    # s_e from a curve at p
    curve = ast.SECurve(0.5, 0.02, 0.9, 10)
    assert ast.runs_needed(0.02, None, 500, p=0.3, curve=curve).s_e == pytest.approx(0.5 * 0.21)
    for bad in (dict(d=0.0, s_e=0.1, n_t=10), dict(d=0.02, s_e=-1, n_t=10),
                dict(d=0.02, s_e=0.1, n_t=10, risk=0.7, criterion="reversal"),
                dict(d=0.02, s_e=0.1, n_t=10, criterion="other"),
                dict(d=0.02, s_e=0.1, n_t=10, alpha=1.5), dict(d=0.02, s_e=0.1, n_t=10, power=0.0)):
        with pytest.raises(ValueError):
            ast.runs_needed(**bad)
    # risk is ignored by the power criterion
    assert ast.runs_needed(0.02, 0.1, 10, risk=0.7).n_r == ast.runs_needed(0.02, 0.1, 10).n_r
    with pytest.raises(ValueError):
        ast.runs_needed(0.02, 0.1, 10, estimand="task_population")
    with pytest.raises(ValueError):
        ast.runs_needed(0.02, None, 10)

@needs_paper
@pytest.mark.parametrize("key", ["swev_rep", "tb2", "deepswe", "tau2_retail"])
def test_runs_needed_and_se_curve_match_paper_code_on_real_data(ext, paper, key):
    _, c3 = paper
    df = _load(key, balanced=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd = ast.run_disagreement(df)
    tbl = ext.se_by_system(ext.cell_table(df))
    fc = ext.fit_se_curve(tbl["mean"].to_numpy(), tbl["s_e"].to_numpy(), tbl["n_cells_used"].to_numpy())
    curve = rd.se_curve()
    assert curve.kappa == pytest.approx(fc["kappa"], rel=1e-9)
    assert curve.kappa_se == pytest.approx(fc["kappa_se"], rel=1e-9)
    assert curve.r2 == pytest.approx(fc["r2"], rel=1e-9)
    p_med = float(tbl["mean"].median())
    assert curve(p_med) == pytest.approx(fc["kappa"] * p_med * (1 - p_med), rel=1e-12)
    n_t = int(df.task_id.nunique())
    lo, hi = curve.s_e_band(p_med)
    assert lo == pytest.approx(max(fc["kappa"] - fc["kappa_se"], 1e-9) * p_med * (1 - p_med), rel=1e-12)
    for d_pts in (1, 2, 3):
        for s_e in (curve(p_med), lo, hi):
            n_paper, cont_paper = c3.nr_star(d_pts, s_e, n_t)
            got = ast.runs_needed(d_pts / 100.0, s_e, n_t)
            assert got.n_r == n_paper and got.n_r_cont == pytest.approx(cont_paper, rel=1e-12)
            n_rev, cont_rev = c3.nr_rev(d_pts, s_e, n_t)
            got_rev = ast.runs_needed(d_pts / 100.0, s_e, n_t, criterion="reversal")
            assert got_rev.n_r == n_rev and got_rev.n_r_cont == pytest.approx(cont_rev, rel=1e-12)
        via_curve = ast.runs_needed(d_pts / 100.0, None, n_t, p=p_med, curve=curve)
        assert via_curve.n_r == c3.nr_star(d_pts, curve(p_med), n_t)[0]


@needs_paper
def test_design_mde_matches_paper_formula(paper):
    _, c3 = paper
    s_e, s_d, n_t = 0.085, 0.02, 500
    for nr in c3.NR_MDE:
        fixed = 100 * c3.ZK * np.sqrt(2 * s_e / (n_t * nr))
        pop = 100 * c3.ZK * np.sqrt((s_d + 2 * s_e / nr) / n_t)
        assert 100 * ast.mde(0.0, s_e, n_t, nr, estimand="fixed_tasks") == pytest.approx(fixed, rel=1e-12)
        assert 100 * ast.mde(s_d, s_e, n_t, nr, estimand="task_population") == pytest.approx(pop, rel=1e-12)


# ----------------------------------------------------------------------------------- plan_second_stage
def _stage1(rng, P=3, T=60, n1=2, hetero=True):
    """Stage-1 frame: half of the tasks have p ~ 0.5, half p ~ 0.02 or 0.98."""
    p = np.where(np.arange(T) % 2 == 0, 0.5, np.where(rng.random(T) < 0.5, 0.02, 0.98)) if hetero else np.full(T, 0.5)
    y = (rng.random((P, T, n1)) < p[None, :, None]).astype(float)
    return df_from_array(y), p


def test_plan_second_stage_budget_bounds_and_columns():
    rng = np.random.default_rng(3)
    df, _ = _stage1(rng, P=3, T=60, n1=2)
    for floor, r_max, budget in ((0, 6, 240), (1, 6, 240), (0, None, 200), (0, 4, 4 * 60)):
        plan = ast.plan_second_stage(df, budget, floor=floor, r_max=r_max, rng=1)
        assert list(plan.columns) == ["system_id", "task_id", "n_stage1", "n_stage2", "n_total", "p_smoothed", "weight"]
        assert len(plan) == 3 * 60
        assert (plan.n_total == plan.n_stage1 + plan.n_stage2).all()
        assert (plan.groupby("system_id").n_total.sum() == budget).all()
        assert (plan.n_stage2 >= floor).all()
        if r_max is not None:
            assert (plan.n_total <= r_max).all()


def test_plan_second_stage_sends_runs_to_disagreeing_cells_and_matches_neyman():
    rng = np.random.default_rng(4)
    df, _ = _stage1(rng, P=2, T=400, n1=2)
    plan = ast.plan_second_stage(df, 5 * 400, r_max=10, smoothing=0.1, rng=2)
    k = df.groupby(["system_id", "task_id"]).score.sum().reset_index(name="k")
    plan = plan.merge(k, on=["system_id", "task_id"])
    mixed = plan[plan.k == 1].n_total.mean()
    same = plan[plan.k != 1].n_total.mean()
    assert mixed > same + 1.5
    # unclipped cells: n_total / weight is constant up to rounding (Neyman); clipped cells sit at the bounds
    free = plan[(plan.n_total > 2) & (plan.n_total < 10)]
    lam = free.n_total.sum() / free.weight.sum()
    assert ((free.n_total - lam * free.weight).abs() <= 1.0 + 1e-9).all()
    # equal stage-1 outcomes in every cell -> equal weights -> uniform allocation (up to a unit)
    flat = df.assign(score=0.0)
    pf = ast.plan_second_stage(flat, 4 * 400, rng=5)
    assert pf.n_total.between(4, 4).all()


def test_plan_second_stage_tie_breaking_is_random_and_seeded():
    rng = np.random.default_rng(5)
    df, _ = _stage1(rng, P=1, T=50, n1=2, hetero=False)
    df = df.assign(score=0.0)  # every cell ties; 125 runs = 2.5 per cell
    a = ast.plan_second_stage(df, 125, rng=11).n_total.to_numpy()
    b = ast.plan_second_stage(df, 125, rng=11).n_total.to_numpy()
    c = ast.plan_second_stage(df, 125, rng=12).n_total.to_numpy()
    assert (a == b).all() and (a != c).any()
    hits = np.mean([ast.plan_second_stage(df, 125, rng=s).n_total.to_numpy() for s in range(200)], axis=0)
    assert hits.min() > 2.3 and hits.max() < 2.7  # no task is favoured by position


def test_plan_second_stage_errors_mapping_and_cap_warning():
    rng = np.random.default_rng(6)
    df, _ = _stage1(rng, P=2, T=10, n1=2)
    with pytest.raises(ValueError, match="stage-1 runs"):
        ast.plan_second_stage(df, 19)
    with pytest.raises(ValueError, match="floor"):
        ast.plan_second_stage(df, 25, floor=1)
    with pytest.raises(ValueError, match="r_max"):
        ast.plan_second_stage(df, 40, r_max=2, floor=1)
    with pytest.raises(ValueError, match="smoothing"):
        ast.plan_second_stage(df, 40, smoothing=0.0)
    with pytest.raises(ValueError, match="no entry"):
        ast.plan_second_stage(df, {"s000": 40})
    plan = ast.plan_second_stage(df, {"s000": 30, "s001": 50}, rng=1)
    assert plan.groupby("system_id").n_total.sum().to_dict() == {"s000": 30, "s001": 50}
    with pytest.warns(UserWarning, match="exceeds"):
        pc = ast.plan_second_stage(df, 100, r_max=4, rng=1)
    assert (pc.n_total == 4).all()


def test_plan_second_stage_minimises_neyman_objective_and_follows_the_smoothed_weights():
    """With the stage-1 weights taken as the cell SDs, the plan's sum_t w_t^2 / n_t beats the uniform allocation of the same budget
    and is within rounding of the continuous optimum (sum w)^2 / budget when no bound binds."""
    rng = np.random.default_rng(8)
    df, _ = _stage1(rng, P=2, T=300, n1=3)
    for a in (0.5, 0.1):
        plan = ast.plan_second_stage(df, 6 * 300, r_max=30, smoothing=a, rng=1)
        for _, g in plan.groupby("system_id"):
            obj = (g.weight**2 / g.n_total).sum()
            assert obj < (g.weight**2).sum() / 6.0 - 1e-6  # uniform allocation of the same budget
            assert obj <= (g.weight.sum()) ** 2 / (6 * 300) * 1.01  # continuous optimum, 1% for integer rounding


def test_plan_second_stage_mean_is_nearly_unbiased_for_binary_runs():
    """Monte Carlo: the number of runs of a cell depends on its stage-1 outcomes, so the pooled mean is not exactly unbiased; the bias is small."""
    rng = np.random.default_rng(7)
    T, R, B = 100, 8, 4 * 100
    p = np.where(np.arange(T) % 2 == 0, 0.5, np.where(np.arange(T) % 4 == 1, 0.05, 0.95))
    est = []
    for _ in range(400):
        y = (rng.random((1, T, R)) < p[None, :, None]).astype(float)
        full = df_from_array(y)
        plan = ast.plan_second_stage(full[full.run_id < 2], B, r_max=R, smoothing=0.5, rng=rng)
        keep = full.merge(plan[["task_id", "n_total"]], on="task_id")
        sub = keep[keep.run_id < keep.n_total]
        est.append(sub.groupby("task_id").score.mean().mean())
    est = np.array(est)
    assert abs(est.mean() - p.mean()) < 4 * est.std(ddof=1) / np.sqrt(len(est)) + 0.003


def test_public_api_exports():
    for name in ("run_disagreement", "reversal_probability", "pair_z", "pair_z_table", "runs_needed", "fit_se_curve", "plan_second_stage"):
        assert name in ast.__all__ and callable(getattr(ast, name))
    assert ast.__version__ == "0.2.1"


def test_readme_quickstart_runs():
    import re

    readme = Path(__file__).resolve().parents[1] / "README.md"
    if not readme.exists():
        pytest.skip("README not present")
    m = re.search(r"## Quickstart.*?```python\n(.*?)```", readme.read_text(encoding="utf-8"), re.S)
    assert m, "quickstart block missing"
    ns: dict = {}
    exec(compile(m.group(1), "README quickstart", "exec"), ns)
    assert 0 < ns["rd"].D < 0.5 and 0 < ns["rd"].rho_run < 1
    assert len([ln for ln in m.group(1).splitlines() if ln.strip()]) <= 12
