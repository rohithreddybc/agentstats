import numpy as np
import pytest
from scipy import stats

import agentstats as ast
from _sim import df_from_array

Z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)


def paired_gauss(rng, T, R, delta, s_delta, s_run, s_base=0.0025, mu=0.5):
    """Systems s000 (a) and s001 (b); true per-task diff = delta + g_t, g_t ~ N(0, s_delta)."""
    base = mu + rng.normal(0, np.sqrt(s_base), T)
    d = delta + rng.normal(0, np.sqrt(s_delta), T)
    ta, tb = base + d / 2, base - d / 2
    y = np.stack([ta, tb])[..., None] + rng.normal(0, np.sqrt(s_run), (2, T, R))
    assert y.min() >= 0 and y.max() <= 1
    return df_from_array(y)


# ------------------------------------------------------------------ paired_diff


def test_signflip_p_exact_matches_brute_force_enumeration():
    rng = np.random.default_rng(0)
    T, R = 9, 3
    y = np.clip(rng.normal(0.5, 0.1, (2, T, R)) + np.array([0.04, 0.0])[:, None, None], 0, 1)
    df = df_from_array(y)
    res = ast.paired_diff(df, "s000", "s001", n_boot=200, rng=0)
    d = y[0].mean(1) - y[1].mean(1)
    obs = abs(d.mean())
    cnt = 0
    for mask in range(2**T):
        s = np.array([1 if (mask >> i) & 1 else -1 for i in range(T)])
        cnt += abs((s * d).mean()) >= obs - 1e-12
    assert res.p_value == pytest.approx(cnt / 2**T)
    assert res.mean_diff == pytest.approx(d.mean())
    assert res.effect_size == pytest.approx(d.mean() / d.std(ddof=1))
    assert res.n_tasks == T


def test_paired_diff_detects_and_ci_brackets():
    rng = np.random.default_rng(1)
    df = paired_gauss(rng, 120, 4, delta=0.03, s_delta=0.0016, s_run=0.0036)
    res = ast.paired_diff(df, "s000", "s001", n_boot=1000, rng=0)
    assert res.lo < res.mean_diff < res.hi and res.lo > 0
    assert res.p_value < 0.01
    sw = ast.paired_diff(df, "s001", "s000", n_boot=1000, rng=0)
    assert sw.mean_diff == pytest.approx(-res.mean_diff)
    assert sw.p_value == pytest.approx(res.p_value, abs=0.02)


@pytest.mark.parametrize("estimand", ["task_population", "fixed_tasks"])
def test_paired_diff_null_calibration(estimand):
    rng = np.random.default_rng(10)
    n_sim, rej, ps = 200, 0, []
    for _ in range(n_sim):
        df = paired_gauss(rng, 40, 4, delta=0.0, s_delta=0.0016 if estimand == "task_population" else 0.0,
                          s_run=0.0036)
        r = ast.paired_diff(df, "s000", "s001", n_boot=200, estimand=estimand, rng=rng)
        ps.append(r.p_value)
        rej += r.p_value < 0.05
    assert 0.02 <= rej / n_sim <= 0.09, rej / n_sim
    assert 0.4 <= np.mean(ps) <= 0.6


def test_paired_diff_uses_shared_tasks_and_errors():
    rng = np.random.default_rng(2)
    df = paired_gauss(rng, 30, 3, 0.02, 0.001, 0.003)
    df = df[~((df.system_id == "s001") & (df.task_id < "t00010"))]  # b misses 10 tasks
    res = ast.paired_diff(df, "s000", "s001", n_boot=100, rng=0)
    assert res.n_tasks == 20
    with pytest.raises(ValueError, match="differ"):
        ast.paired_diff(df, "s000", "s000")
    with pytest.raises(ValueError, match="not found"):
        ast.paired_diff(df, "s000", "zzz")
    disjoint = df[((df.system_id == "s000") & (df.task_id < "t00015"))
                  | ((df.system_id == "s001") & (df.task_id >= "t00015"))]
    with pytest.raises(ValueError, match="share no tasks"):
        ast.paired_diff(disjoint, "s000", "s001")


# ------------------------------------------------------------------ mde


def test_mde_formulas_by_hand():
    m = ast.mde(0.0016, 0.0036, 100, 5, alpha=0.05, power=0.8)
    assert m == pytest.approx(Z * np.sqrt((0.0016 + 2 * 0.0036 / 5) / 100))
    f = ast.mde(0.0016, 0.0036, 100, 5, estimand="fixed_tasks")
    assert f == pytest.approx(Z * np.sqrt(2 * 0.0036 / (100 * 5)))
    assert f < m
    # sigma2_delta_task is ignored for fixed tasks
    assert ast.mde(5.0, 0.0036, 100, 5, estimand="fixed_tasks") == pytest.approx(f)
    arr = ast.mde(0.0016, 0.0036, np.array([50, 100, 200]), 5)
    assert arr.shape == (3,) and (np.diff(arr) < 0).all()
    with pytest.raises(ValueError):
        ast.mde(-1, 0.1, 10, 1)
    with pytest.raises(ValueError):
        ast.mde(0.1, 0.1, 10, 0)


def test_mde_from_df_recovers_variances():
    rng = np.random.default_rng(3)
    s_delta, s_run = 0.0016, 0.0036
    df = paired_gauss(rng, 600, 6, delta=0.0, s_delta=s_delta, s_run=s_run)
    r = ast.mde_from_df(df, "s000", "s001")
    assert r.sigma2_run == pytest.approx(s_run, rel=0.1)
    assert r.sigma2_delta_task == pytest.approx(s_delta, rel=0.15)
    assert (r.n_t, r.n_r) == (600, 6)
    assert r.mde == pytest.approx(ast.mde(r.sigma2_delta_task, r.sigma2_run, 600, 6))
    r2 = ast.mde_from_df(df, "s000", "s001", n_t=100, n_r=3, estimand="fixed_tasks")
    assert r2.mde == pytest.approx(ast.mde(0, r.sigma2_run, 100, 3, estimand="fixed_tasks"))


def test_paired_tests_at_the_mde_reject_at_nominal_power_task_population():
    rng = np.random.default_rng(4)
    T, R, s_delta, s_run = 100, 5, 0.0016, 0.0036
    m = ast.mde(s_delta, s_run, T, R)
    n_sim, rej = 400, 0
    for _ in range(n_sim):
        df = paired_gauss(rng, T, R, delta=m, s_delta=s_delta, s_run=s_run)
        r = ast.paired_diff(df, "s000", "s001", n_boot=200, rng=rng)
        rej += r.p_value < 0.05
    power = rej / n_sim
    assert 0.7 <= power <= 0.9, power


def test_paired_tests_at_the_mde_reject_at_nominal_power_fixed_tasks():
    rng = np.random.default_rng(5)
    T, R, s_run = 50, 4, 0.0036
    m = ast.mde(0.0, s_run, T, R, estimand="fixed_tasks")
    n_sim, rej = 300, 0
    for _ in range(n_sim):
        df = paired_gauss(rng, T, R, delta=m, s_delta=0.0, s_run=s_run)
        r = ast.paired_diff(df, "s000", "s001", n_boot=200, estimand="fixed_tasks", rng=rng)
        rej += r.p_value < 0.05
    power = rej / n_sim
    assert 0.7 <= power <= 0.9, power


# ------------------------------------------------------------------ claim_check


def test_claim_check_verdicts():
    kw = dict(n_t=100, n_r=5, sigma2_delta_task=0.0016, sigma2_run=0.0036)
    m = ast.mde(0.0016, 0.0036, 100, 5)
    se = m / Z
    big = ast.claim_check(0.60 + m * 1.2, 0.60, **kw)
    assert big.verdict == "clear" and big.p_value < 0.005
    assert big.mde == pytest.approx(m) and big.se == pytest.approx(se)
    assert big.z == pytest.approx((m * 1.2) / se)
    # between 1.96 se and 2.81 se (p between 0.005 and 0.05): marginal, never "underpowered"
    mid = ast.claim_check(0.60 + 2.2 * se, 0.60, **kw)
    assert mid.verdict == "marginal" and 0.005 <= mid.p_value < 0.05
    small = ast.claim_check(0.60 + 1.0 * se, 0.60, **kw)
    assert small.verdict == "not distinguishable" and small.p_value > 0.05
    neg = ast.claim_check(0.60, 0.60 + m * 1.2, **kw)
    assert neg.verdict == "clear" and neg.diff < 0
    assert small.p_value == pytest.approx(2 * stats.norm.sf(1.0))
    f = ast.claim_check(0.61, 0.60, estimand="fixed_tasks", **kw)
    assert f.se < se
    with pytest.raises(ValueError):
        ast.claim_check(0.6, 0.5, 10, 1, 0.0, 0.0)


def test_claim_check_verdict_depends_on_p_only_and_thresholds():
    kw = dict(n_t=100, n_r=5, sigma2_delta_task=0.0016, sigma2_run=0.0036)
    se = ast.claim_check(0.6, 0.6 - 1e-9, **kw).se
    z_clear, z_05 = stats.norm.isf(0.005 / 2), stats.norm.isf(0.05 / 2)
    assert ast.claim_check(0.6 + (z_clear + 0.01) * se, 0.6, **kw).verdict == "clear"
    assert ast.claim_check(0.6 + (z_clear - 0.01) * se, 0.6, **kw).verdict == "marginal"
    assert ast.claim_check(0.6 + (z_05 + 0.01) * se, 0.6, **kw).verdict == "marginal"
    assert ast.claim_check(0.6 + (z_05 - 0.01) * se, 0.6, **kw).verdict == "not distinguishable"
    # MDE is reported but does not drive the verdict: a gap above the MDE has p < 0.005 anyway,
    # and the verdict set has no "underpowered" / "exceeds MDE" / "within noise".
    seen = {ast.claim_check(0.6 + k * se, 0.6, **kw).verdict for k in np.linspace(0, 5, 51)}
    assert seen == {"clear", "marginal", "not distinguishable"}
    # custom thresholds
    r = ast.claim_check(0.6 + 2.2 * se, 0.6, alpha=0.1, clear_alpha=0.04, **kw)
    assert r.verdict == "clear"
    with pytest.raises(ValueError):
        ast.claim_check(0.6, 0.5, alpha=0.01, clear_alpha=0.05, **kw)


def test_paired_diff_schemes():
    rng = np.random.default_rng(21)
    df = paired_gauss(rng, 100, 4, delta=0.02, s_delta=0.0004, s_run=0.0049)
    cl = ast.paired_diff(df, "s000", "s001", n_boot=3000, rng=0)  # default cluster
    tl = ast.paired_diff(df, "s000", "s001", n_boot=3000, rng=0, scheme="two_level")
    assert cl.mean_diff == tl.mean_diff and cl.p_value == pytest.approx(tl.p_value, abs=0.03)
    assert (tl.hi - tl.lo) > (cl.hi - cl.lo)
    with pytest.raises(ValueError, match="scheme"):
        ast.paired_diff(df, "s000", "s001", scheme="x")
