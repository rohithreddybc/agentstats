"""Simulation helpers shared by the tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def df_from_array(y, benchmark="b", sys_prefix="s", task_prefix="t") -> pd.DataFrame:
    """Long frame from a (P, T, R) array of scores."""
    P, T, R = y.shape
    s, t, r = np.meshgrid(np.arange(P), np.arange(T), np.arange(R), indexing="ij")
    return pd.DataFrame(
        {
            "benchmark": benchmark,
            "task_id": np.char.add(task_prefix, np.char.zfill(t.ravel().astype(str), 5)),
            "system_id": np.char.add(sys_prefix, np.char.zfill(s.ravel().astype(str), 3)),
            "run_id": r.ravel().astype(np.int64),
            "score": np.asarray(y, dtype=float).ravel(),
        }
    )


def standardise(z, var):
    """Centre ``z`` and scale it to sample variance (ddof=1) exactly ``var``."""
    z = np.asarray(z, dtype=float)
    z = z - z.mean()
    return z / z.std(ddof=1) * np.sqrt(var)


def simulate_crossed(rng, P, T, R, s_p, s_t, s_pt, s_e, mu=0.5, kind="gaussian"):
    """Balanced p x t x r:(pt) data with known variance components.

    The system and task effects are centred and scaled to have sample variance exactly ``s_p`` /
    ``s_t``: with only 8 systems the sampling SD of the realised system variance (about
    ``s_p * sqrt(2/7)``) would otherwise dwarf the 0.003 tolerance of the spec.  The
    interaction and run noise are random.

    ``kind="gaussian"`` returns Gaussian scores (not confined to [0, 1]) and the nominal
    components.  ``kind="bernoulli"`` draws uniform effects, sets the cell probability
    ``pi_pt = mu + a_p + b_t + c_pt`` (asserted to lie in [0, 1]) and returns 0/1 scores; the
    true residual variance is then ``mean(pi (1 - pi))``.
    """
    if kind == "gaussian":
        a = standardise(rng.normal(size=P), s_p)
        b = standardise(rng.normal(size=T), s_t)
        c = rng.normal(0.0, np.sqrt(s_pt), (P, T))
        y = mu + a[:, None, None] + b[None, :, None] + c[..., None]
        y = y + rng.normal(0.0, np.sqrt(s_e), (P, T, R))
        return y, {"p": s_p, "t": s_t, "pt": s_pt, "e": s_e}
    a = standardise(rng.uniform(-1, 1, P), s_p)
    b = standardise(rng.uniform(-1, 1, T), s_t)
    c = rng.uniform(-np.sqrt(3 * s_pt), np.sqrt(3 * s_pt), (P, T))
    pi = mu + a[:, None] + b[None, :] + c
    assert pi.min() >= 0 and pi.max() <= 1, "cell probabilities out of range; shrink components"
    y = (rng.random((P, T, R)) < pi[..., None]).astype(float)
    return y, {"p": s_p, "t": s_t, "pt": s_pt, "e": float(np.mean(pi * (1 - pi)))}


def bernoulli_tasks_df(rng, P, T, R, skills, sd_task=0.8, sd_pt=0.0, benchmark="b"):
    """Systems with logit-scale ``skills`` on shared tasks; returns a long 0/1 frame."""
    b = rng.normal(0.0, sd_task, T)
    logit = np.asarray(skills, dtype=float)[:, None] + b[None, :] + rng.normal(0.0, sd_pt, (P, T))
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random((P, T, R)) < p[..., None]).astype(float)
    return df_from_array(y, benchmark=benchmark)
