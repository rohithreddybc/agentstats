import warnings
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import agentstats as ast
from agentstats.ranking import _ordinal_ranks, _tau_from_ranks
from _sim import bernoulli_tasks_df


def test_tau_matches_scipy_kendall():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=12)
    reps = ref + rng.normal(0, 0.7, (50, 12))
    tau = _tau_from_ranks(_ordinal_ranks(reps), _ordinal_ranks(ref))
    for i in range(50):
        assert tau[i] == pytest.approx(stats.kendalltau(reps[i], ref)[0])


def test_well_separated_systems_give_tau_near_one():
    rng = np.random.default_rng(1)
    df = bernoulli_tasks_df(rng, 6, 150, 3, skills=np.linspace(-1.5, 1.5, 6), sd_task=0.8)
    rs = ast.rank_stability(df, n_boot=1000, top_k=1, rng=0)
    assert rs.tau_mean > 0.97
    assert rs.rank_probs.iloc[0, 0] > 0.97  # best system is rank 1
    assert rs.topk_change_prob < 0.03
    assert rs.pairwise_flip.to_numpy().max() < 0.05
    assert list(rs.systems) == list(rs.means.sort_values(ascending=False).index)
    assert np.allclose(rs.rank_probs.sum(axis=1), 1) and np.allclose(rs.rank_probs.sum(axis=0), 1)
    assert len(rs.tau) == 1000 and rs.estimand == "task_population"
    pf = rs.pairwise_flip.to_numpy()
    assert np.allclose(pf, pf.T) and np.allclose(np.diag(pf), 0)


def test_identical_systems_are_maximally_unstable():
    """For systems with identical true scores the observed ranking is itself noise, so the
    bootstrap replicate ranking agrees with it only partially: expected Kendall tau is
    (2/pi) * arcsin(1/sqrt(2)) = 0.5 (replicate = observed + independent noise of equal size),
    not 0.  The spec's 'near 0' would require an independent reference ranking."""
    taus, top1_change, p1 = [], [], []
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)
        df = bernoulli_tasks_df(rng, 10, 150, 3, skills=np.zeros(10), sd_task=0.8)
        rs = ast.rank_stability(df, n_boot=400, rng=seed)
        taus.append(rs.tau_mean)
        top1_change.append(rs.topk_change_prob)
        p1.append(rs.rank_probs[1].max())
    assert 0.38 <= np.mean(taus) <= 0.62, np.mean(taus)
    assert np.mean(top1_change) > 0.5
    assert np.mean(p1) < 0.6  # no system is clearly first


def test_identical_systems_vs_independent_reference_tau_near_zero():
    """Kendall tau between rankings of two independent datasets of identical systems is ~0."""
    taus = []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        m1 = rng.normal(size=10)
        m2 = rng.normal(size=10)
        taus.append(stats.kendalltau(m1, m2)[0])
    assert abs(np.mean(taus)) < 0.08


def test_schemes_and_estimands_run_and_two_level_is_noisier():
    rng = np.random.default_rng(3)
    df = bernoulli_tasks_df(rng, 4, 60, 4, skills=[0.0, 0.1, 0.2, 0.3], sd_task=1.0)
    a = ast.rank_stability(df, n_boot=600, rng=0)  # default: cluster
    b = ast.rank_stability(df, n_boot=600, scheme="two_level", rng=0)
    c = ast.rank_stability(df, n_boot=600, estimand="fixed_tasks", rng=0)
    assert b.tau_mean <= a.tau_mean + 0.02  # extra (double-counted) run noise -> less stable
    assert all(0 <= r.tau_mean <= 1 for r in (a, b, c))
    with pytest.raises(ValueError, match="scheme"):
        ast.rank_stability(df, scheme="nope")


def test_rank_stability_errors_and_ragged_support():
    rng = np.random.default_rng(4)
    df = bernoulli_tasks_df(rng, 3, 30, 4, skills=[0, 0.5, 1.0])
    with pytest.raises(ValueError, match="top_k"):
        ast.rank_stability(df, top_k=3)
    with pytest.raises(ValueError, match="every system"):
        ast.rank_stability(df[~((df.system_id == "s001") & (df.task_id == "t00000"))])
    with pytest.raises(ValueError, match="at least 2 systems"):
        ast.rank_stability(df[df.system_id == "s000"])
    ragged = df[~((df.system_id == "s002") & (df.run_id >= 2))]
    rs = ast.rank_stability(ragged, n_boot=100, rng=0)
    assert rs.n_boot == 100
    a = ast.rank_stability(df, n_boot=100, rng=7)
    b = ast.rank_stability(df, n_boot=100, rng=7)
    assert np.array_equal(a.tau, b.tau)


def test_top_k_set_change():
    rng = np.random.default_rng(5)
    df = bernoulli_tasks_df(rng, 5, 80, 3, skills=[0, 0.2, 0.4, 2.0, 2.1], sd_task=0.5)
    rs = ast.rank_stability(df, n_boot=800, top_k=2, rng=0)
    assert rs.topk_change_prob < 0.15  # {s003, s004} well separated from the rest
    rs1 = ast.rank_stability(df, n_boot=800, top_k=1, rng=0)
    assert rs1.topk_change_prob > rs.topk_change_prob  # 2.0 vs 2.1 is hard to call


def test_subsample_curve_halves_and_vs_full():
    rng = np.random.default_rng(6)
    df = bernoulli_tasks_df(rng, 6, 120, 12, skills=np.linspace(-0.6, 0.6, 6), sd_task=0.8)
    curve = ast.subsample_curve(df, [1, 6], [20, 60, 120], n_rep=60, rng=0)
    assert list(curve.columns) == [
        "n_runs", "n_tasks", "n_rep", "reference", "mean_tau", "mean_halfwidth", "top1_flip_rate"
    ]
    assert len(curve) == 6 and (curve.reference == "half_vs_half").all()
    c6 = curve[curve.n_runs == 6].sort_values("n_tasks")
    assert (np.diff(c6.mean_tau) > 0).all() and (np.diff(c6.mean_halfwidth) < 0).all()
    small = curve[(curve.n_runs == 1) & (curve.n_tasks == 20)].iloc[0]
    big = curve[(curve.n_runs == 6) & (curve.n_tasks == 120)].iloc[0]
    assert small.mean_halfwidth > big.mean_halfwidth and small.mean_tau < big.mean_tau
    assert big.mean_tau < 1.0  # two independent halves never agree by construction
    again = ast.subsample_curve(df, [1, 6], [20, 60, 120], n_rep=60, rng=0)
    pd.testing.assert_frame_equal(curve, again)
    with pytest.raises(ValueError, match="disjoint"):
        ast.subsample_curve(df, [7], [20])  # 2 * 7 > 12
    with pytest.raises(ValueError, match="exceeds"):
        ast.subsample_curve(df, [2], [500])
    # old metric: full size reproduces the reference exactly (tau = 1 by construction)
    old = ast.subsample_curve(df, [12], [120], n_rep=20, rng=0, vs_full=True)
    assert old.reference.iloc[0] == "vs_full"
    assert old.mean_tau.iloc[0] == 1.0 and old.top1_flip_rate.iloc[0] == 0.0
    with pytest.raises(ValueError, match="exceeds"):
        ast.subsample_curve(df, [13], [20], vs_full=True)


def test_subsample_curve_vs_full_is_biased_upward_for_identical_systems():
    taus_half, taus_full = [], []
    for seed in range(8):
        rng = np.random.default_rng(300 + seed)
        df = bernoulli_tasks_df(rng, 8, 100, 10, skills=np.zeros(8), sd_task=0.8)
        taus_half.append(ast.subsample_curve(df, [1], [100], n_rep=60, rng=seed).mean_tau[0])
        taus_full.append(
            ast.subsample_curve(df, [1], [100], n_rep=60, rng=seed, vs_full=True).mean_tau[0]
        )
    assert abs(np.mean(taus_half)) < 0.05
    assert np.mean(taus_full) > 0.2


# ----------------------------------------------------------------------- rerun agreement


def test_rerun_agreement_well_separated_systems():
    rng = np.random.default_rng(1)
    df = bernoulli_tasks_df(rng, 6, 150, 10, skills=np.linspace(-1.5, 1.5, 6), sd_task=0.8)
    r = ast.rerun_agreement(df, 2, n_rep=300, rng=0)
    assert r.tau_mean > 0.95 and r.p_top1_same > 0.95 and r.p_topk_same[3] > 0.95
    assert set(r.p_topk_same) == {1, 3, 5}
    assert r.tau_quantiles.index.tolist() == [0.025, 0.25, 0.5, 0.75, 0.975]
    assert r.n_cells_skipped == 0 and r.n_tasks_used == 150 and r.n_tasks_dropped == 0
    pf = r.pairwise_flip.to_numpy()
    assert np.allclose(pf, pf.T) and np.allclose(np.diag(pf), 0) and pf.max() < 0.05
    m = ast.rerun_agreement_model(df, 2, n_rep=300, rng=0)
    assert m.tau_mean > 0.95


def test_rerun_agreement_identical_systems_tau_zero_and_top1_chance():
    """Independent-rerun agreement of identical systems is 0 (unlike the bootstrap vs full data,
    which gives 0.5).  Per-dataset values scatter (halves share the same finite data), so average
    over datasets."""
    P = 8
    taus, top1 = [], []
    for seed in range(25):
        rng = np.random.default_rng(1000 + seed)
        df = bernoulli_tasks_df(rng, P, 150, 10, skills=np.zeros(P), sd_task=0.8)
        r = ast.rerun_agreement(df, 1, n_rep=300, rng=seed)
        taus.append(r.tau_mean)
        top1.append(r.p_top1_same)
    assert abs(np.mean(taus)) < 0.05, np.mean(taus)
    assert abs(np.mean(top1) - 1 / P) < 0.03, np.mean(top1)


def test_rerun_agreement_skips_and_reports_short_cells():
    rng = np.random.default_rng(2)
    df = bernoulli_tasks_df(rng, 3, 40, 6, skills=[0.0, 0.5, 1.0])
    short = df[~((df.system_id == "s001") & (df.task_id < "t00010") & (df.run_id >= 3))]
    r = ast.rerun_agreement(short, 3, n_rep=50, rng=0)  # needs 6 runs per cell
    assert r.n_cells_skipped == 10 and r.n_tasks_dropped == 10 and r.n_tasks_used == 30
    with pytest.raises(ValueError, match="no task"):
        ast.rerun_agreement(df, 4, n_rep=10)
    a = ast.rerun_agreement(df, 2, n_rep=50, rng=5)
    b = ast.rerun_agreement(df, 2, n_rep=50, rng=5)
    assert np.array_equal(a.tau, b.tau)
    assert list(a.p_topk_same) == [1]  # k = 3, 5 are not below the 3 systems


def test_rerun_model_agrees_with_empirical_where_both_apply():
    rng = np.random.default_rng(1)
    diffs = []
    for sd_pt in (0.3, 0.0, 0.3):
        df = bernoulli_tasks_df(
            rng, 8, 150, 12, skills=np.linspace(-0.3, 0.3, 8), sd_task=0.8, sd_pt=sd_pt
        )
        for n_runs in (1, 2, 3):
            e = ast.rerun_agreement(df, n_runs, n_rep=800, rng=0)
            m = ast.rerun_agreement_model(df, n_runs, n_rep=800, rng=0)
            diffs.append(m.tau_mean - e.tau_mean)
            assert abs(m.tau_mean - e.tau_mean) < 0.05, (sd_pt, n_runs, m.tau_mean, e.tau_mean)
    assert abs(np.mean(diffs)) < 0.03


def test_rerun_model_identical_systems_and_beyond_half_runs():
    P = 8
    taus, top1 = [], []
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        df = bernoulli_tasks_df(rng, P, 150, 10, skills=np.zeros(P), sd_task=0.8)
        m = ast.rerun_agreement_model(df, 1, n_rep=300, rng=seed)
        taus.append(m.tau_mean)
        top1.append(m.p_top1_same)
        assert m.info["model"] == "empirical_bayes_binomial"
        assert set(m.info["shrinkage"]) == {"system", "task", "interaction"}
    assert abs(np.mean(taus)) < 0.05 and abs(np.mean(top1) - 1 / P) < 0.03
    # n_runs beyond what disjoint halves allow: the empirical version refuses, the model works
    rng = np.random.default_rng(3)
    df = bernoulli_tasks_df(rng, 6, 100, 10, skills=np.linspace(-0.5, 0.5, 6), sd_task=0.8)
    with pytest.raises(ValueError):
        ast.rerun_agreement(df, 8, n_rep=10)
    lo = ast.rerun_agreement_model(df, 2, n_rep=400, rng=0).tau_mean
    hi = ast.rerun_agreement_model(df, 40, n_rep=400, rng=0).tau_mean
    assert hi > lo
    few = ast.rerun_agreement_model(df, 5, n_tasks=20, n_rep=400, rng=0).tau_mean
    many = ast.rerun_agreement_model(df, 5, n_tasks=300, n_rep=400, rng=0).tau_mean
    assert many > few
    with pytest.raises(ValueError, match="binary"):
        ast.rerun_agreement_model(df.assign(score=df.score * 0.5), 2)


def test_rerun_model_calibrated_flag_and_warning():
    P = 8
    df = bernoulli_tasks_df(np.random.default_rng(11), P, 150, 10, skills=np.zeros(P), sd_task=0.8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a calibrated call must not warn
        m = ast.rerun_agreement_model(df, 1, n_rep=200, rng=0)
    assert m.info["calibrated"] is True and abs(m.info["null_tau"]) <= m.info["null_tolerance"]
    with pytest.warns(UserWarning, match="not calibrated under the null"):
        far = ast.rerun_agreement_model(df, 60, n_rep=200, rng=0)
    assert far.info["calibrated"] is False and far.info["null_tau"] > 0.05
    skipped = ast.rerun_agreement_model(df, 60, n_rep=200, rng=0, check_null=False)
    assert skipped.info["calibrated"] is None and skipped.info["null_tau"] is None
    # the flag does not change the estimate (the null check uses its own generator)
    assert skipped.tau_mean == far.tau_mean


def test_rerun_model_from_variance_components():
    from agentstats.variance import VarianceComponents

    def vc(s_p, s_pt=0.01):
        d = dict(sigma2_p=s_p, sigma2_t=0.05, sigma2_pt=s_pt, sigma2_e=0.03)
        tot = sum(d.values())
        return VarianceComponents(
            **d, prop_p=s_p / tot, prop_t=0.05 / tot, prop_pt=s_pt / tot, prop_e=0.03 / tot,
            n_p=8, n_t=100, n_r=10,
        )

    # identical systems = no system effect AND no system x task interaction (the interaction is
    # shared by both evaluations of the same tasks, so it is a real difference on these tasks)
    null = ast.rerun_agreement_model(vc(0.0, 0.0), 3, n_rep=6000, rng=0)
    assert abs(null.tau_mean) < 0.05 and abs(null.p_top1_same - 1 / 8) < 0.02
    shared = ast.rerun_agreement_model(vc(0.0), 3, n_rep=3000, rng=0)
    assert 0.2 < shared.tau_mean < 0.45  # fixed-task interaction alone: tau ~ 1/3
    sep = ast.rerun_agreement_model(vc(0.5), 3, n_rep=1000, rng=0)
    assert sep.tau_mean > 0.95
    assert ast.rerun_agreement_model(vc(0.002), 3, n_rep=50, rng=0).info["calibrated"] is True
    more = ast.rerun_agreement_model(vc(0.002), 20, n_rep=2000, rng=0).tau_mean
    less = ast.rerun_agreement_model(vc(0.002), 1, n_rep=2000, rng=0).tau_mean
    assert more > less
