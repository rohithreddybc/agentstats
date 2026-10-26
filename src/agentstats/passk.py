"""pass@k (Chen et al. 2021) and pass^k (tau-bench style) estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln

from ._utils import check_int
from .data import build_tensor, prep


def _nck(n, c, k):
    n = np.asarray(n)
    c = np.asarray(c)
    k = np.asarray(k)
    n, c, k = np.broadcast_arrays(n, c, k)
    for name, a in (("n", n), ("c", c), ("k", k)):
        if not np.issubdtype(a.dtype, np.number) or not np.all(np.mod(a, 1) == 0):
            raise ValueError(f"{name} must be integer-valued")
    n, c, k = n.astype(np.int64), c.astype(np.int64), k.astype(np.int64)
    if (c < 0).any() or (c > n).any():
        raise ValueError("need 0 <= c <= n")
    if (k < 1).any() or (k > n).any():
        raise ValueError("need 1 <= k <= n")
    return n, c, k


def _log_comb(n, k):
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def _scalar_if_0d(a: np.ndarray):
    return float(a) if a.ndim == 0 else a


def pass_at_k(n, c, k):
    """Unbiased pass@k estimator: ``1 - C(n-c, k) / C(n, k)`` (Chen et al. 2021).

    ``n`` attempts of which ``c`` succeeded; the probability that at least one of ``k`` attempts
    drawn without replacement succeeds.  Vectorised over broadcastable ``n, c, k``.
    """
    n, c, k = _nck(n, c, k)
    ok = (n - c) >= k
    m = np.where(ok, n - c, k)
    fail = np.where(ok, np.exp(_log_comb(m, k) - _log_comb(n, k)), 0.0)
    return _scalar_if_0d(1.0 - fail)


def pass_hat_k(n, c, k):
    """pass^k = ``C(c, k) / C(n, k)``: probability that all ``k`` drawn attempts succeed.

    Unbiased estimator of ``p^k`` for per-task success probability ``p`` (tau-bench).
    Vectorised over broadcastable ``n, c, k``.
    """
    n, c, k = _nck(n, c, k)
    ok = c >= k
    m = np.where(ok, c, k)
    return _scalar_if_0d(np.where(ok, np.exp(_log_comb(m, k) - _log_comb(n, k)), 0.0))


def system_pass_k(df: pd.DataFrame, system_id: str, k: int, kind: str = "pass_at_k") -> float:
    """Mean over tasks of per-task pass@k (``kind="pass_at_k"``) or pass^k (``"pass_hat_k"``).

    Scores must be binary (0/1) and every task needs at least ``k`` runs (use
    :func:`agentstats.balance`).  Estimand: point estimate of the task-averaged quantity; for
    uncertainty bootstrap with :func:`agentstats.bootstrap_ci` on a per-task transform.
    """
    if kind not in ("pass_at_k", "pass_hat_k"):
        raise ValueError("kind must be 'pass_at_k' or 'pass_hat_k'")
    k = check_int(k, "k")
    df = prep(df)
    if system_id not in set(df["system_id"]):
        raise ValueError(f"system_id {system_id!r} not found in data")
    t = build_tensor(df, systems=[system_id])
    sc = df.loc[df["system_id"] == system_id, "score"].to_numpy()
    if not np.isin(sc, (0.0, 1.0)).all():
        raise ValueError("pass@k / pass^k require binary 0/1 scores")
    n = t.counts[0]
    if n.min() < k:
        raise ValueError(
            f"k={k} exceeds the number of runs on some task (min {int(n.min())}); lower k or balance"
        )
    c = np.rint(t.x[0].sum(-1)).astype(np.int64)
    f = pass_at_k if kind == "pass_at_k" else pass_hat_k
    return float(np.mean(f(n, c, k)))
