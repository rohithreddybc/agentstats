# agentstats

Statistics for LLM agent benchmark results: how noisy is a score, how many runs and tasks a
claim needs, and whether a reported ranking or gain survives resampling.

Runtime dependencies: numpy, pandas, scipy. Python >= 3.10.

## Install

```
pip install agentstats
```

## Quickstart

Run disagreement `D`, the run share `rho_run` and the number of runs per task needed to separate
a 2-point gap (the quantities Q1 and Q4 of the accompanying paper), on simulated data:

```python
import numpy as np, pandas as pd, agentstats as ag

rng = np.random.default_rng(0)
P, T, R = 8, 300, 5                                    # systems, tasks, runs per cell
p_cell = np.clip(rng.beta(1, 2, T) + 0.03 * np.arange(P)[:, None] + rng.normal(0, 0.1, (P, T)), 0, 1)  # (P, T)
s, t, r = np.indices((P, T, R))
df = pd.DataFrame({"benchmark": "demo", "task_id": t.ravel().astype(str), "system_id": s.ravel().astype(str),
                   "run_id": r.ravel(), "score": rng.binomial(1, p_cell[s, t]).ravel().astype(float)})
rd = ag.run_disagreement(df)                           # Q1: D = 2 s_e, rho_run = s_e / (s_pt + s_e)
print(f"D = {rd.D:.3f}, rho_run = {rd.rho_run:.3f}")
print(ag.runs_needed(0.02, rd.se_curve()(0.4), T))     # Q4: runs per task so the design MDE (80% power) is <= 2 points
```

## Data

Long format, one row per attempt: `benchmark, task_id, system_id, run_id, score` (score in
[0, 1]; binary pass/fail is 0/1), plus optional `model, harness, cost_usd, tokens_in,
tokens_out, wall_s, source`.

## Estimands

Every function states one of two estimands:

- `fixed_tasks`: uncertainty of the score on *this* benchmark's task set; only runs are random.
  The bootstrap resamples centred residuals within each (system, task) cell, inflated by
  `sqrt(R/(R-1))` (adjusted residual resampling), which removes the `(R-1)/R` variance deficit of
  plain run resampling. Cells with a single run contribute no variance (warning if > 10%).
- `task_population`: uncertainty about the population the tasks were drawn from; tasks and
  runs are both random (default). `scheme="cluster"` (default) resamples tasks and keeps each
  task's runs, so run noise is counted once; `scheme="two_level"` also resamples runs within
  tasks and is conservative (run noise counted twice).

## Usage

```python
import numpy as np
import pandas as pd
import agentstats as ast

rng = np.random.default_rng(0)
rows = []
for system, skill in [("model-a", 0.62), ("model-b", 0.58)]:
    task_difficulty = np.random.default_rng(1).normal(0, 0.8, 200)  # same tasks for both
    for t, d in enumerate(task_difficulty):
        p = 1 / (1 + np.exp(-(skill * 2 - 1 + d)))
        for r in range(5):
            rows.append(("demo", f"task{t}", system, r, float(rng.random() < p)))
df = pd.DataFrame(rows, columns=["benchmark", "task_id", "system_id", "run_id", "score"])

ast.validate(df)
print(ast.bootstrap_ci(df, "model-a", rng=0))          # CI(mean, lo, hi, ...)
print(ast.paired_diff(df, "model-a", "model-b", n_boot=2000, rng=0))
bal, info = ast.balance(df, min_runs=5)                # balanced design
vc = ast.variance_components(bal)                      # sigma2_p, sigma2_t, sigma2_pt, sigma2_e
print(ast.d_study(vc, n_t=[100, 200], n_r=[1, 5]).e_rho2)
print(ast.mde_from_df(df, "model-a", "model-b"))       # minimum detectable difference
print(ast.rank_stability(df, n_boot=2000, rng=0).tau_mean)   # biased upward, see Notes
print(ast.rerun_agreement(df, n_runs=2, rng=0).tau_mean)       # independent-rerun agreement
print(ast.report(df, n_boot=1000, rng=0))              # markdown reporting card
```

Command line:

```
agentstats report results.csv --benchmark swe-bench-verified --seed 0
```

(`.parquet` input needs an optional parquet engine such as `pyarrow`, which is not a dependency.)

## Named quantities

One function per quantity of the paper, each with its estimand in the docstring.

| quantity | call | estimand |
|---|---|---|
| Q1 run disagreement `D` (unbiased `2k(R-k)/(R(R-1))` per cell, ragged runs allowed) and run share `rho_run = s_e/(s_pt+s_e)` | `run_disagreement(df)` | `fixed_tasks` |
| run variance curve `s_e(p) = kappa p (1-p)` | `fit_se_curve(m, s_e, w)`, `run_disagreement(df).se_curve()` | `fixed_tasks` |
| fixed-task reliability `(s_p + s_pt/n_t) / (s_p + s_pt/n_t + s_e/(n_t n_r))` | `d_study(vc, n_t, n_r).fixed_tasks` | `fixed_tasks` |
| Q3 reversal probability `Phi(-abs(z)/sqrt(2))` and pair-level `z` | `reversal_probability(z)`, `pair_z(df, a, b)`, `pair_z_table(df)` | `fixed_tasks` |
| Q4 runs needed `n_r*(d)`: smallest `n_r` whose design MDE (two-sided `alpha`, `power`) is `<= d`; `criterion="reversal"` gives the weaker `n_r_rev(d)` with `Phi(-d/SE) <= risk` | `runs_needed(d, s_e, n_t, estimand=..., criterion="power", alpha=0.05, power=0.8)` | both |
| Q4 design MDE | `mde(...)` | both |

`claim_check` returns the verdicts `"clear"` (p < 0.005), `"marginal"` (0.005 <= p < 0.05) and
`"not distinguishable"` (p >= 0.05). The MDE is reported as a property of the design and never
enters the verdict.

`rerun_agreement_model` (data-frame input) is not calibrated under the null beyond small
`n_runs`; check `info["calibrated"]` before using its values.

## Ranking stability: which function answers which question

- `rerun_agreement(df, n_runs)`: *how often would an independent rerun change the ranking?*
  Two disjoint sets of `n_runs` runs per cell are ranked and compared with each other (estimand
  `fixed_tasks`): Kendall tau, P(top-1 identical), P(top-k set identical), pairwise flip matrix.
  Identical systems give tau = 0 and P(top-1 identical) = 1/n_systems.
  `rerun_agreement_model(df_or_vc, n_runs, n_tasks=None)` is a parametric version (empirical-Bayes
  binomial model for binary data, Gaussian model for variance components) for `n_runs` beyond half
  the available runs.
- `rank_stability(df)` bootstraps and compares each replicate ranking with the *full-data*
  ranking. That reference is itself noisy, so its tau is **biased upward** (identical systems give
  about 0.5, not 0). Use it for P(rank = r) and pairwise-flip probabilities, not as rerun agreement.
- `subsample_curve(...)` uses disjoint halves by default. `vs_full=True` restores the old
  subsample-vs-full-data metric, which is **biased upward** for the same reason (and equals 1 at the
  full size by construction).

## Notes

- Variance components use ANOVA expected-mean-squares estimators for the balanced crossed
  system x task design with runs nested in cells; negative estimates are truncated at 0 and
  flagged. On 0/1 scores this is a linear-probability decomposition.
- Bootstraps are vectorised with index arrays.
- All random functions take `rng` as a `numpy.random.Generator`, an int seed, or `None`.

## Citation

If you use this package, please cite the accompanying paper (placeholder until publication):

```
Bellibatlu, R. R. and Singh, M. How Many Runs Is Enough? Rerun Noise, Rank Reversals and Run Budgets for LLM Agent Benchmarks. 2026. https://github.com/rohithreddybc/agentstats
```

## Licence

MIT.
