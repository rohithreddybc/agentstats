"""Markdown reporting card for one benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import as_rng
from .bootstrap import bootstrap_ci
from .compare import claim_check, mde_from_df
from .data import balance, prep, system_scores
from .ranking import rank_stability, rerun_agreement, rerun_agreement_model
from .variance import d_study, variance_components, variance_components_single


def _fmt(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def report(df: pd.DataFrame, n_boot: int = 2000, level: float = 0.95, rng=None) -> str:
    """Markdown reporting card for a single benchmark.

    Sections: scores with CIs (both estimands), variance components, D-study for
    ``n_r in {1, 3, 5, 10}`` at the observed number of tasks, rank stability, MDE between
    adjacent systems, and a checklist of what authors should report.  Scores, rank stability and
    MDEs use the tasks common to all systems; variance components and the D-study use the
    design balanced to the smallest run count (the balancing counts are printed).  Sections that
    cannot be computed (e.g. one run per task) say so instead of being skipped silently.
    """
    df = prep(df)
    rng = as_rng(rng)
    bench = df["benchmark"].iloc[0]
    systems = sorted(df["system_id"].unique())
    common = set.intersection(*[set(df.loc[df["system_id"] == s, "task_id"]) for s in systems])
    if not common:
        raise ValueError("the systems share no common tasks; cannot report")
    cdf = df[df["task_id"].isin(common)]
    n_tasks_all = df["task_id"].nunique()
    runs_per_cell = cdf.groupby(["system_id", "task_id"]).size()
    min_runs = int(runs_per_cell.min())
    max_runs = int(runs_per_cell.max())
    L: list[str] = [f"# Benchmark reporting card: {bench}", ""]
    L.append(
        f"{len(systems)} systems, {len(common)} tasks common to all systems"
        f" (of {n_tasks_all}), {min_runs}-{max_runs} runs per system-task cell,"
        f" {len(cdf)} attempts."
    )
    if len(common) < n_tasks_all:
        L.append(f"Note: {n_tasks_all - len(common)} task(s) not evaluated by every system were excluded.")
    L.append("")

    # 1. scores
    L += ["## Scores", ""]
    sc = system_scores(cdf).set_index("system_id")
    rows = []
    for s in sc.sort_values("mean", ascending=False).index:
        ci_p = bootstrap_ci(cdf, s, level, n_boot, "task_population", rng)
        if max_runs >= 2:
            ci_f = bootstrap_ci(cdf, s, level, n_boot, "fixed_tasks", rng)
            fixed = f"[{_fmt(ci_f.lo)}, {_fmt(ci_f.hi)}]"
        else:
            fixed = "n/a (1 run)"
        rows.append([s, _fmt(sc.loc[s, "mean"]), f"[{_fmt(ci_p.lo)}, {_fmt(ci_p.hi)}]", fixed])
    pct = int(round(level * 100))
    L.append(
        _md_table(
            ["system", "mean", f"{pct}% CI task_population", f"{pct}% CI fixed_tasks"], rows
        )
    )
    L += [
        "",
        "task_population: tasks and runs random (generalise to the task population). "
        "fixed_tasks: only run-to-run noise on this task set.",
        "",
    ]

    # 2. variance components + 3. D-study
    L += ["## Variance components", ""]
    vc = None
    if min_runs < 2:
        L += ["Not available: every cell has a single run, so run-to-run variance is not identifiable.", ""]
    else:
        bal, info = balance(cdf, min_runs)
        L.append(
            f"Balanced to {min_runs} runs per cell: {info['n_tasks_kept']} tasks kept, "
            f"{info['rows_truncated']} extra attempts truncated."
        )
        L.append("")
        if len(systems) >= 2 and info["n_tasks_kept"] >= 2:
            vc = variance_components(bal)
            fr = vc.to_frame()
            L.append(
                _md_table(
                    ["component", "variance", "share", "truncated at 0"],
                    [
                        [r.component, _fmt(r.variance, 5), f"{100 * r.proportion:.1f}%", str(r.truncated)]
                        for r in fr.itertuples()
                    ],
                )
            )
            L += ["", "Linear-probability decomposition if scores are 0/1.", ""]
        else:
            s0 = systems[0]
            v1 = variance_components_single(bal, s0)
            L.append(
                _md_table(
                    ["system", "sigma2_task", "sigma2_run", "ICC(1)"],
                    [[s0, _fmt(v1.sigma2_task, 5), _fmt(v1.sigma2_run, 5), _fmt(v1.icc)]],
                )
            )
            L.append("")

    L += ["## D-study (reliability of system rankings)", ""]
    if vc is None:
        L += ["Not available (needs >= 2 systems, >= 2 tasks and >= 2 runs per cell).", ""]
    else:
        grid = [1, 3, 5, 10]
        ds = d_study(vc, vc.n_t, grid)
        L.append(
            _md_table(
                ["runs per task", "E rho2", "Phi"],
                [[g, _fmt(a), _fmt(b)] for g, a, b in zip(grid, ds.e_rho2, ds.phi)],
            )
        )
        L += ["", f"At the observed {vc.n_t} tasks (task_population).", ""]

    # 4. rank stability
    L += ["## Rank stability (bootstrap vs full-data ranking)", ""]
    rs = None
    if len(systems) < 2:
        L += ["Not available (single system).", ""]
    else:
        rs = rank_stability(cdf, n_boot, "task_population", 1, rng)
    if rs is not None:
        L.append(
            _md_table(
                ["rank", "system", "mean", "P(rank 1)", "P(observed rank)"],
                [
                    [i + 1, s, _fmt(rs.means[s]), _fmt(rs.rank_probs.loc[s, 1]), _fmt(rs.rank_probs.loc[s, i + 1])]
                    for i, s in enumerate(rs.systems)
                ],
            )
        )
        L += [
            "",
            f"Mean Kendall tau vs the full-data ranking: {_fmt(rs.tau_mean)}; "
            f"P(top-1 changes): {_fmt(rs.topk_change_prob)} "
            f"({rs.n_boot} bootstrap resamples, task_population).",
            "",
        ]

    # 4b. rerun agreement
    L += ["## Rerun agreement (would an independent rerun change the ranking?)", ""]
    if len(systems) < 2 or min_runs < 2:
        L += ["Not available (needs >= 2 systems and >= 2 runs per cell).", ""]
    else:
        rrows = []
        half = min_runs // 2
        ra = rerun_agreement(cdf, half, n_rep=500, rng=rng)
        rrows.append(("split-half, same tasks", ra, half))
        if cdf["score"].isin([0.0, 1.0]).all():
            rm = rerun_agreement_model(cdf, min_runs, n_rep=500, rng=rng)
            rrows.append(("model, full run count", rm, min_runs))
        tab = []
        for name, r, nr in rrows:
            q = r.tau_quantiles
            tab.append(
                [name, nr, _fmt(r.tau_mean), f"[{_fmt(q[0.025])}, {_fmt(q[0.975])}]",
                 _fmt(r.p_top1_same), _fmt(r.p_topk_same.get(3, float("nan")))]
            )
        L.append(
            _md_table(
                ["version", "runs per evaluation", "mean tau", "tau 95% range", "P(top-1 same)", "P(top-3 set same)"],
                tab,
            )
        )
        L += [
            "",
            "Two independent evaluations of the same tasks are compared with each other (fixed_tasks). "
            "Unlike the bootstrap rank stability above, which compares with the noisy full-data "
            "ranking and is biased upward, identical systems give tau = 0 here.",
            "",
        ]

    # 5. adjacent MDE
    L += ["## Minimum detectable difference between adjacent systems", ""]
    if rs is None or min_runs < 2:
        L += ["Not available (needs >= 2 systems and >= 2 runs per cell).", ""]
    else:
        rows = []
        for a, b in zip(rs.systems[:-1], rs.systems[1:]):
            m = mde_from_df(cdf, a, b)
            gap = float(rs.means[a] - rs.means[b])
            cc = claim_check(
                rs.means[a], rs.means[b], m.n_t, m.n_r, m.sigma2_delta_task, m.sigma2_run
            )
            rows.append([f"{a} vs {b}", _fmt(gap), _fmt(m.mde), _fmt(cc.p_value, 4), cc.verdict])
        L.append(
            _md_table(["pair", "observed gap", "MDE (80% power, alpha 0.05)", "approx p", "verdict"], rows)
        )
        L += ["", "Variances estimated from these data (task_population).", ""]

    # 6. checklist
    L += ["## What authors should report", ""]
    L.append("- [ ] Number of tasks and number of runs per task, for every system.")
    L.append("- [ ] Estimand: fixed task set or generalisation to a task population.")
    L.append("- [ ] Confidence intervals for scores (bootstrap over tasks and runs), not just means.")
    L.append("- [ ] Variance components (task, system x task, run) or the run-to-run SD.")
    L.append("- [ ] Paired differences with CI and test for every claimed improvement.")
    L.append("- [ ] Minimum detectable difference of the design next to every claimed gap.")
    L.append("- [ ] Stability of the ranking (e.g. P(rank 1)) when ranking more than two systems.")
    L.append("- [ ] Seeds, harness version, model snapshot and sampling settings.")
    if min_runs < 2:
        L.append("- [!] Only one run per cell: run-to-run noise cannot be separated; add repeats.")
    elif min_runs < 5:
        L.append(f"- [!] Only {min_runs} runs per cell: variance components will be imprecise.")
    if len(common) < 100:
        L.append(f"- [!] {len(common)} tasks: small task sample; task-level uncertainty dominates.")
    return "\n".join(L) + "\n"
