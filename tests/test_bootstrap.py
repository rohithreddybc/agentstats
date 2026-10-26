import numpy as np
import pytest
from scipy import stats

import agentstats as ast
from _sim import df_from_array


def beta_task_data(rng, T, R, a=2.0, b=2.0):
    """One system; per-task success probability ~ Beta(a, b); returns (df, mean p of these tasks)."""
    p = rng.beta(a, b, T)
    y = (rng.random((1, T, R)) < p[None, :, None]).astype(float)
    return df_from_array(y), float(p.mean())


def test_ci_basic_properties_and_reproducibility():
    df, _ = beta_task_data(np.random.default_rng(0), 60, 4)
    a = ast.bootstrap_ci(df, "s000", n_boot=500, rng=42)
    b = ast.bootstrap_ci(df, "s000", n_boot=500, rng=42)
    assert a == b
    assert a.lo < a.mean < a.hi and a.level == 0.95 and a.estimand == "task_population"
    assert a.n_boot == 500
    assert a.mean == pytest.approx(df["score"].mean())  # balanced: mean of cell means
    c = ast.bootstrap_ci(df, "s000", n_boot=500, rng=np.random.default_rng(42))
    assert c == a
    assert ast.bootstrap_ci(df, "s000", n_boot=500, rng=43) != a


def test_fixed_tasks_narrower_than_task_population():
    df, _ = beta_task_data(np.random.default_rng(1), 80, 5)
    tp = ast.bootstrap_ci(df, "s000", n_boot=2000, estimand="task_population", rng=0)
    ft = ast.bootstrap_ci(df, "s000", n_boot=2000, estimand="fixed_tasks", rng=0)
    assert (ft.hi - ft.lo) < (tp.hi - tp.lo)
    assert ft.estimand == "fixed_tasks"


def test_ci_width_matches_bootstrap_theory_cluster_and_two_level():
    """Plug-in theory.  Cluster bootstrap: Var* = var0(task means)/T (run noise already inside
    the task means).  Two-level: Var* = var0(task means)/T + mean_t(var0_t)/(R T), with var0
    using divisor n; the second term double counts run noise."""
    T, R = 400, 5
    df, _ = beta_task_data(np.random.default_rng(2), T, R)
    y = df["score"].to_numpy().reshape(T, R)
    stage1 = y.mean(1).var(ddof=0) / T
    stage2 = y.var(axis=1, ddof=0).mean() / (R * T)
    z = stats.norm.ppf(0.975)
    cl = ast.bootstrap_ci(df, "s000", n_boot=6000, rng=0)  # default scheme is cluster
    tl = ast.bootstrap_ci(df, "s000", n_boot=6000, rng=0, scheme="two_level")
    assert (cl.hi - cl.lo) / 2 == pytest.approx(z * np.sqrt(stage1), rel=0.05)
    assert (tl.hi - tl.lo) / 2 == pytest.approx(z * np.sqrt(stage1 + stage2), rel=0.05)
    assert (tl.hi - tl.lo) > 1.1 * (cl.hi - cl.lo)  # strongly noise-dominated data: +17% width


def test_ragged_runs_supported():
    df, _ = beta_task_data(np.random.default_rng(3), 50, 6)
    df = df[~((df.task_id.isin(["t00000", "t00001"])) & (df.run_id >= 2))]
    ci = ast.bootstrap_ci(df, "s000", n_boot=300, rng=0)
    cm = df.groupby("task_id")["score"].mean()
    assert ci.mean == pytest.approx(cm.mean())
    assert ci.lo < ci.mean < ci.hi


@pytest.mark.parametrize(
    "estimand,scheme", [("task_population", "cluster"), ("task_population", "two_level"), ("fixed_tasks", "x")]
)
def test_bootstrap_matches_naive_loop_distribution(estimand, scheme):
    """The vectorised engine and a plain Python resampling loop give the same interval."""
    T, R = 40, 4
    df, _ = beta_task_data(np.random.default_rng(4), T, R)
    y = df["score"].to_numpy().reshape(T, R)
    rng = np.random.default_rng(0)
    k = np.sqrt(R / (R - 1))
    reps = []
    for _ in range(8000):
        if estimand == "fixed_tasks":  # adjusted residual resampling of runs
            runs = rng.integers(0, R, (T, R))
            m = y.mean(1)
            reps.append((m + k * (np.take_along_axis(y, runs, 1).mean(1) - m)).mean())
        elif scheme == "cluster":
            reps.append(y[rng.integers(0, T, T)].mean())
        else:
            tasks = rng.integers(0, T, T)
            runs = rng.integers(0, R, (T, R))
            reps.append(np.take_along_axis(y[tasks], runs, 1).mean())
    kw = {} if estimand == "fixed_tasks" else {"scheme": scheme}
    ci = ast.bootstrap_ci(df, "s000", n_boot=8000, estimand=estimand, rng=1, **kw)
    naive = np.quantile(reps, [0.025, 0.975])
    # Monte Carlo error of each quantile is about 0.003, so compare width (rel 8%) and centre
    assert (ci.hi - ci.lo) == pytest.approx(naive[1] - naive[0], rel=0.08)
    assert (ci.hi + ci.lo) / 2 == pytest.approx(naive.mean(), abs=0.004)


def test_fixed_tasks_variance_correction_and_singleton_warning():
    T, R = 300, 3
    df, _ = beta_task_data(np.random.default_rng(8), T, R)
    y = df["score"].to_numpy().reshape(T, R)
    # unbiased variance of the mean: mean_t(s2_t)/(R T) (+ nothing else under fixed tasks)
    se = np.sqrt(y.var(axis=1, ddof=1).mean() / (R * T))
    ci = ast.bootstrap_ci(df, "s000", n_boot=6000, estimand="fixed_tasks", rng=0)
    assert (ci.hi - ci.lo) / 2 == pytest.approx(1.96 * se, rel=0.05)
    # > 10% single-run cells -> warning; <= 10% -> none
    few = df[~((df.run_id > 0) & (df.task_id < "t00040"))]  # 40 of 300 tasks (13%) single run
    with pytest.warns(UserWarning, match="single run"):
        ast.bootstrap_ci(few, "s000", n_boot=50, estimand="fixed_tasks", rng=0)
    ok = df[~((df.run_id > 0) & (df.task_id < "t00020"))]  # 6.7%
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ast.bootstrap_ci(ok, "s000", n_boot=50, estimand="fixed_tasks", rng=0)


def test_ci_input_errors():
    df, _ = beta_task_data(np.random.default_rng(0), 10, 3)
    with pytest.raises(ValueError, match="not found"):
        ast.bootstrap_ci(df, "zzz")
    with pytest.raises(ValueError, match="estimand"):
        ast.bootstrap_ci(df, "s000", estimand="bogus")
    with pytest.raises(ValueError, match="level"):
        ast.bootstrap_ci(df, "s000", level=1.5)
    with pytest.raises(ValueError, match="scheme"):
        ast.bootstrap_ci(df, "s000", scheme="bogus")
    with pytest.raises(ValueError, match="rng"):
        ast.bootstrap_ci(df, "s000", rng="seed")
    with pytest.raises(ValueError, match="n_boot"):
        ast.bootstrap_ci(df, "s000", n_boot=3)
    one_run = df[df.run_id == 0]
    with pytest.raises(ValueError, match="2 runs"):
        ast.bootstrap_ci(one_run, "s000", estimand="fixed_tasks", n_boot=50)
    ast.bootstrap_ci(one_run, "s000", estimand="task_population", n_boot=50, rng=0)


def test_task_population_coverage_200_datasets():
    """Coverage of the 95% task_population (cluster) CI over 200 simulated datasets must be in
    [0.91, 0.98]."""
    rng = np.random.default_rng(2026)
    hits = 0
    n_sim = 200
    for _ in range(n_sim):
        df, _ = beta_task_data(rng, 100, 5)  # Beta(2,2): population mean exactly 0.5
        ci = ast.bootstrap_ci(df, "s000", n_boot=1000, estimand="task_population", rng=rng)
        hits += ci.lo <= 0.5 <= ci.hi
    cover = hits / n_sim
    assert 0.91 <= cover <= 0.98, cover


@pytest.mark.parametrize("R", [3, 5])
def test_fixed_tasks_coverage_with_adjusted_residuals(R):
    """Truth = mean of the realised per-task probabilities.  Plain run resampling covers only
    ~88% (R=3) / ~92% (R=5); the adjusted residual resampling must land in [0.93, 0.97]."""
    rng = np.random.default_rng(1234 + R)
    hits = 0
    n_sim = 1000
    for _ in range(n_sim):
        df, truth = beta_task_data(rng, 100, R)
        ci = ast.bootstrap_ci(df, "s000", n_boot=1000, estimand="fixed_tasks", rng=rng)
        hits += ci.lo <= truth <= ci.hi
    cover = hits / n_sim
    print(f"fixed_tasks coverage R={R}: {cover:.3f}")
    assert 0.93 <= cover <= 0.97, cover


def test_iqm_known_values():
    assert ast.iqm(np.arange(1, 9)) == pytest.approx(4.5)  # mean of 3..6
    x = np.random.default_rng(0).normal(size=100)
    assert ast.iqm(x) == pytest.approx(stats.trim_mean(x, 0.25))
    assert ast.iqm([5.0]) == 5.0
    # n = 5: lo = 1.25, hi = 3.75 -> 0.75*x1 + x2 + 0.75*x3 over 2.5
    s = np.array([1.0, 2.0, 4.0, 7.0, 100.0])
    assert ast.iqm(s[::-1]) == pytest.approx((0.75 * 2 + 4 + 0.75 * 7) / 2.5)
    with pytest.raises(ValueError):
        ast.iqm([])
    with pytest.raises(ValueError):
        ast.iqm([1.0, np.nan])


def test_iqm_ci():
    rng = np.random.default_rng(0)
    y = np.clip(rng.normal(0.5, 0.15, (1, 60, 5)) + rng.normal(0, 0.1, (1, 60, 1)), 0, 1)
    df = df_from_array(y)
    tp = ast.iqm_ci(df, "s000", n_boot=500, rng=0)
    ft = ast.iqm_ci(df, "s000", n_boot=500, estimand="fixed_tasks", rng=0)
    assert tp.mean == pytest.approx(ast.iqm(y)) == pytest.approx(ft.mean)
    assert tp.lo < tp.mean < tp.hi
    assert (ft.hi - ft.lo) < (tp.hi - tp.lo)
    assert ast.iqm_ci(df, "s000", n_boot=500, rng=0) == tp
    with pytest.raises(ValueError, match="equal run counts"):
        ast.iqm_ci(df.drop(index=df.index[3]), "s000", n_boot=50)
