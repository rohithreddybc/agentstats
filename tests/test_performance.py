import time

import numpy as np
import pytest

import agentstats as ast
from _sim import df_from_array


def big_df(P=20, T=500, R=10, seed=0):
    rng = np.random.default_rng(seed)
    p = 1 / (1 + np.exp(-(rng.normal(0, 0.3, (P, 1)) + rng.normal(0, 1, (1, T)))))
    return df_from_array((rng.random((P, T, R)) < p[..., None]).astype(float))


@pytest.mark.slow
@pytest.mark.parametrize("kw", [{}, {"scheme": "two_level"}, {"estimand": "fixed_tasks"}],
                         ids=["cluster", "two_level", "fixed_tasks"])
def test_10k_resamples_500_tasks_10_runs_20_systems_under_30s(kw):
    df = big_df()
    t0 = time.perf_counter()
    rs = ast.rank_stability(df, n_boot=10_000, rng=0, **kw)
    elapsed = time.perf_counter() - t0
    print(f"\nrank_stability {kw or 'default'}, 10,000 resamples x 20 x 500 x 10: {elapsed:.1f} s")
    assert rs.n_boot == 10_000
    assert elapsed < 30.0, elapsed


@pytest.mark.slow
def test_10k_bootstrap_ci_for_every_system_under_30s():
    df = big_df(seed=1)
    t0 = time.perf_counter()
    for s in sorted(df.system_id.unique()):
        ast.bootstrap_ci(df, s, n_boot=10_000, rng=0, estimand="fixed_tasks")
    elapsed = time.perf_counter() - t0
    print(f"\n20 x bootstrap_ci(fixed_tasks, n_boot=10,000): {elapsed:.1f} s")
    assert elapsed < 30.0, elapsed
