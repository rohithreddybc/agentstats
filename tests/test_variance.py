import numpy as np
import pandas as pd
import pytest

import agentstats as ast
from agentstats.variance import VarianceComponents
from _sim import bernoulli_tasks_df, df_from_array, simulate_crossed

SPEC = dict(s_p=0.02, s_t=0.05, s_pt=0.01, s_e=0.03)


def close(est, true):
    """Spec tolerance: within 15% relative or 0.003 absolute."""
    return abs(est - true) <= max(0.15 * true, 0.003)


def check_recovery(vc, truth):
    for k in ("p", "t", "pt", "e"):
        est = getattr(vc, f"sigma2_{k}")
        assert close(est, truth[k]), f"sigma2_{k}: est {est:.5f} vs true {truth[k]:.5f}"


def test_ems_recovers_gaussian_components():
    rng = np.random.default_rng(20261017)
    y, truth = simulate_crossed(rng, 8, 300, 10, **SPEC)
    vc = ast.variance_components_array(y)
    assert (vc.n_p, vc.n_t, vc.n_r) == (8, 300, 10)
    check_recovery(vc, truth)
    assert not any(vc.truncated.values())
    assert vc.prop_p + vc.prop_t + vc.prop_pt + vc.prop_e == pytest.approx(1.0)


def test_ems_recovers_bernoulli_components_via_dataframe():
    rng = np.random.default_rng(7)
    y, truth = simulate_crossed(
        rng, 8, 300, 10, s_p=0.004, s_t=0.02, s_pt=0.004, s_e=0.0, kind="bernoulli"
    )
    df = df_from_array(y)
    ast.validate(df)
    vc = ast.variance_components(df)
    check_recovery(vc, truth)


def test_dataframe_route_matches_array_route():
    rng = np.random.default_rng(3)
    y = np.clip(0.5 + rng.normal(0, 0.1, (4, 20, 3)), 0, 1)
    a = ast.variance_components(df_from_array(y))
    b = ast.variance_components_array(y)
    assert a.sigma2_p == pytest.approx(b.sigma2_p) and a.sigma2_e == pytest.approx(b.sigma2_e)


def test_ems_mean_squares_expectations_unbiased_over_replications():
    """Independent of luck: average of the untruncated estimators over many replications
    (plain random effects, no standardisation) must equal the true components."""
    rng = np.random.default_rng(11)
    P, T, R = 5, 40, 3
    truth = dict(p=0.02, t=0.05, pt=0.01, e=0.03)
    est = {k: [] for k in truth}
    for _ in range(1500):
        a = rng.normal(0, np.sqrt(truth["p"]), P)
        b = rng.normal(0, np.sqrt(truth["t"]), T)
        c = rng.normal(0, np.sqrt(truth["pt"]), (P, T))
        y = a[:, None, None] + b[None, :, None] + c[..., None] + rng.normal(0, np.sqrt(truth["e"]), (P, T, R))
        vc = ast.variance_components_array(y)
        for k in truth:
            est[k].append(vc.raw[k])
    for k, v in truth.items():
        x = np.asarray(est[k])
        se = x.std(ddof=1) / np.sqrt(len(x))
        assert abs(x.mean() - v) < 4 * se, (k, x.mean(), v, se)


def test_negative_estimates_truncated_and_flagged():
    rng = np.random.default_rng(5)
    # pure noise: all structural components ~ 0, some raw estimates negative
    flagged = 0
    for _ in range(30):
        vc = ast.variance_components_array(rng.normal(0, 1, (3, 6, 2)))
        for k in ("p", "t", "pt"):
            raw = vc.raw[k]
            assert getattr(vc, f"sigma2_{k}") == max(raw, 0.0)
            assert vc.truncated[k] == (raw < 0)
            flagged += vc.truncated[k]
    assert flagged > 0


def test_variance_components_requires_balance():
    rng = np.random.default_rng(0)
    df = df_from_array(rng.random((3, 6, 3)))
    ragged = df.drop(index=df.index[5])
    with pytest.raises(ValueError, match="balanced"):
        ast.variance_components(ragged)
    with pytest.raises(ValueError):
        ast.variance_components_array(rng.random((1, 6, 3)))
    with pytest.raises(ValueError):
        ast.variance_components_array(rng.random((3, 6, 1)))
    bal, _ = ast.balance(ragged, 3)
    assert ast.variance_components(bal).n_t < 6


def test_single_system_one_way_icc():
    rng = np.random.default_rng(1)
    T, R = 400, 8
    s_task, s_run = 0.02, 0.03
    b = rng.normal(0, np.sqrt(s_task), T)
    y = 0.5 + b[None, :, None] + rng.normal(0, np.sqrt(s_run), (1, T, R))
    # the data frame route needs scores in [0, 1]: shrink towards 0.5, variances scale by k^2
    k = 0.5
    ys = 0.5 + k * (y - 0.5)
    assert ys.min() > 0 and ys.max() < 1
    df = df_from_array(ys)
    v = ast.variance_components_single(df, "s000")
    assert v.n_t == T and v.n_r == R and not v.truncated_task
    assert v.sigma2_run == pytest.approx(s_run * k**2, rel=0.1)
    assert v.sigma2_task == pytest.approx(s_task * k**2, rel=0.2)
    assert v.icc == pytest.approx(v.sigma2_task / (v.sigma2_task + v.sigma2_run))
    # exact check against a direct computation
    cm = ys[0].mean(1)
    ms_b = R * ((cm - cm.mean()) ** 2).sum() / (T - 1)
    ms_w = ((ys[0] - cm[:, None]) ** 2).sum() / (T * (R - 1))
    assert v.sigma2_run == pytest.approx(ms_w)
    assert v.sigma2_task == pytest.approx((ms_b - ms_w) / R)


def test_single_system_errors():
    df = df_from_array(np.random.default_rng(0).random((2, 5, 3)))
    with pytest.raises(ValueError, match="not found"):
        ast.variance_components_single(df, "nope")
    ragged = df.drop(index=df.index[2])
    with pytest.raises(ValueError, match="same number of runs"):
        ast.variance_components_single(ragged, "s000")


def _vc(**kw):
    d = dict(sigma2_p=0.02, sigma2_t=0.05, sigma2_pt=0.01, sigma2_e=0.03)
    d.update(kw)
    tot = sum(d.values()) or 1.0
    return VarianceComponents(
        **d, prop_p=d["sigma2_p"] / tot, prop_t=d["sigma2_t"] / tot, prop_pt=d["sigma2_pt"] / tot,
        prop_e=d["sigma2_e"] / tot, n_p=8, n_t=300, n_r=10,
    )


def test_d_study_hand_computed_and_limits():
    vc = _vc()
    ds = ast.d_study(vc, n_t=100, n_r=5)
    # rel error: 0.01/100 + 0.03/500 = 1.6e-4; abs error adds 0.05/100 = 5e-4
    assert float(ds.e_rho2) == pytest.approx(0.02 / (0.02 + 1.6e-4))
    assert float(ds.phi) == pytest.approx(0.02 / (0.02 + 6.6e-4))
    big = ast.d_study(vc, n_t=1e9, n_r=1e9)
    assert float(big.e_rho2) == pytest.approx(1.0, abs=1e-6)
    assert float(big.phi) == pytest.approx(1.0, abs=1e-6)
    # monotone and vectorised over a grid
    grid = ast.d_study(vc, n_t=np.array([[10], [100], [1000]]), n_r=np.array([1, 3, 10]))
    assert grid.e_rho2.shape == (3, 3)
    assert (np.diff(grid.e_rho2, axis=0) > 0).all() and (np.diff(grid.e_rho2, axis=1) > 0).all()
    assert (grid.phi <= grid.e_rho2).all()
    assert (grid.fixed_tasks >= grid.e_rho2).all() and (grid.fixed_tasks < 1).all()
    # fixed-task reliability: s_p_fix = s_p + s_pt/n_t over s_p_fix + s_e/(n_t n_r)
    sp_fix = 0.02 + 0.01 / 100
    assert float(ds.fixed_tasks) == pytest.approx(sp_fix / (sp_fix + 0.03 / 500))
    assert float(big.fixed_tasks) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        ast.d_study(vc, n_t=0, n_r=1)
    with pytest.raises(ValueError, match="zero"):
        ast.d_study(_vc(sigma2_p=0, sigma2_t=0, sigma2_pt=0, sigma2_e=0), 10, 10)


def _single_df(seed=0, T=200, R=6):
    rng = np.random.default_rng(seed)
    return bernoulli_tasks_df(rng, 1, T, R, skills=[0.0], sd_task=1.0)


def test_trials_needed_fixed_tasks_is_minimal():
    df = _single_df()
    v = ast.variance_components_single(df, "s000")
    target = 0.01
    n_r = ast.trials_needed(df, "s000", target, estimand="fixed_tasks")
    z = 1.959963984540054

    def hw(n):
        return z * np.sqrt(v.sigma2_run / (v.n_t * n))

    assert hw(n_r) <= target < hw(n_r - 1) or n_r == 1
    with pytest.raises(ValueError, match="max_runs"):
        ast.trials_needed(df, "s000", 1e-4, estimand="fixed_tasks", max_runs=10)


def test_trials_needed_task_population_table():
    df = _single_df()
    v = ast.variance_components_single(df, "s000")
    target = 0.03
    tab = ast.trials_needed(df, "s000", target, estimand="task_population")
    assert isinstance(tab, pd.DataFrame) and len(tab) >= 2
    assert list(tab.columns) == ["n_t", "n_r", "n_attempts", "halfwidth"]
    z = 1.959963984540054
    for r in tab.itertuples():
        var = v.sigma2_task / r.n_t + v.sigma2_run / (r.n_t * r.n_r)
        assert z * np.sqrt(var) == pytest.approx(r.halfwidth)
        assert r.halfwidth <= target + 1e-12
        if r.n_r > 1:
            var1 = v.sigma2_task / r.n_t + v.sigma2_run / (r.n_t * (r.n_r - 1))
            assert z * np.sqrt(var1) > target
    # with few tasks the target is unreachable no matter how many runs
    with pytest.raises(ValueError, match="not reachable"):
        ast.trials_needed(df, "s000", target, estimand="task_population", n_t_grid=[2, 3])
