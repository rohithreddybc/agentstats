from itertools import combinations
from math import comb

import numpy as np
import pytest

import agentstats as ast
from _sim import df_from_array


def enumerate_probs(n, c, k):
    """Exact probabilities by enumerating every k-subset of n attempts with c successes."""
    items = [1] * c + [0] * (n - c)
    subsets = list(combinations(range(n), k))
    all_ok = sum(all(items[i] for i in s) for s in subsets) / len(subsets)
    any_ok = sum(any(items[i] for i in s) for s in subsets) / len(subsets)
    return any_ok, all_ok


def test_pass_hat_k_and_pass_at_k_equal_enumeration():
    for n in range(1, 8):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                any_ok, all_ok = enumerate_probs(n, c, k)
                assert ast.pass_hat_k(n, c, k) == pytest.approx(all_ok, abs=1e-12), (n, c, k)
                assert ast.pass_at_k(n, c, k) == pytest.approx(any_ok, abs=1e-12), (n, c, k)


def test_pass_at_k_closed_form_chen_2021():
    for n, c, k in [(200, 30, 1), (200, 30, 10), (200, 30, 100), (100, 3, 50), (50, 50, 5)]:
        closed = 1.0 - comb(n - c, k) / comb(n, k)
        assert ast.pass_at_k(n, c, k) == pytest.approx(closed, rel=1e-10)
        # numerically stable product form from the paper
        prod = 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)) if n - c >= k else 1.0
        assert ast.pass_at_k(n, c, k) == pytest.approx(prod, rel=1e-10)
    assert ast.pass_at_k(10, 0, 3) == 0.0
    assert ast.pass_at_k(10, 8, 3) == 1.0  # n - c < k


def test_vectorised_and_validation():
    n = np.array([5, 5, 10])
    c = np.array([0, 3, 10])
    out = ast.pass_at_k(n, c, 2)
    assert out.shape == (3,) and out[0] == 0 and out[2] == 1
    grid = ast.pass_hat_k(10, np.arange(11)[:, None], np.arange(1, 6)[None, :])
    assert grid.shape == (11, 5)
    assert ast.pass_hat_k(4, 4, 4) == 1.0 and ast.pass_hat_k(4, 3, 4) == 0.0
    for bad in [(5, 6, 1), (5, -1, 1), (5, 2, 0), (5, 2, 6), (5.5, 2, 1), (5, 2, 1.5)]:
        with pytest.raises(ValueError):
            ast.pass_at_k(*bad)
        with pytest.raises(ValueError):
            ast.pass_hat_k(*bad)


def test_pass_hat_k_unbiased_for_p_to_the_k():
    rng = np.random.default_rng(0)
    p, n, k = 0.7, 8, 3
    c = rng.binomial(n, p, 200_000)
    assert ast.pass_hat_k(n, c, k).mean() == pytest.approx(p**k, abs=0.003)
    assert ast.pass_at_k(n, c, k).mean() == pytest.approx(1 - (1 - p) ** k, abs=0.003)


def test_system_pass_k():
    y = np.zeros((2, 3, 4))
    y[0, 0, :] = [1, 1, 0, 0]  # c=2 of 4
    y[0, 1, :] = [1, 1, 1, 1]
    y[0, 2, :] = 0
    y[1, :, :] = 1
    df = df_from_array(y)
    exp_at = np.mean([ast.pass_at_k(4, 2, 2), 1.0, 0.0])
    exp_hat = np.mean([ast.pass_hat_k(4, 2, 2), 1.0, 0.0])
    assert ast.system_pass_k(df, "s000", 2) == pytest.approx(exp_at)
    assert ast.system_pass_k(df, "s000", 2, kind="pass_hat_k") == pytest.approx(exp_hat)
    assert ast.system_pass_k(df, "s001", 4, kind="pass_hat_k") == 1.0
    with pytest.raises(ValueError, match="exceeds"):
        ast.system_pass_k(df, "s000", 5)
    y2 = y.copy()
    y2[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        ast.system_pass_k(df_from_array(y2), "s000", 2)
