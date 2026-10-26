# Changelog

## 0.2.0 - 2026-10-19

Added
- `run_disagreement`: run disagreement rate `D` (unbiased `2k(R-k)/(R(R-1))` per cell, ragged runs) per release and per system, and the run share `rho_run = s_e/(s_pt+s_e)`.
- `d_study(...)` returns `fixed_tasks`, the fixed-task reliability.
- `reversal_probability`, `pair_z`, `pair_z_table`: reversal probability `Phi(-|z|/sqrt(2))` and pair-level `z` with the fixed-task standard error.
- `runs_needed`, `fit_se_curve`, `SECurve`: runs needed for a given gap and the `s_e(p) = kappa p (1-p)` curve. `runs_needed(..., criterion="power", alpha=0.05, power=0.8)` returns the smallest `n_r` whose design MDE (two-sided test) is at most `d`; this is the default and the headline `n_r*(d)`. `criterion="reversal"` returns the smaller `n_r_rev(d)` with `Phi(-d / SE) <= risk` (the `risk` argument is used only by this criterion). `RunsNeeded` also reports `criterion`, `alpha`, `power` and the multiplier `z`.
- `rerun_agreement_model` reports `info["calibrated"]` (null check on identical systems) and warns when it fails; `check_null=False` skips it.
- PyPI packaging metadata, README quickstart, this changelog.

Changed
- `claim_check` verdicts are now `"clear"` (p < 0.005), `"marginal"` (0.005 <= p < alpha) and `"not distinguishable"` (p >= alpha). The old `"exceeds MDE"`, `"underpowered"` and `"within noise"` are removed: classifying a result by its observed effect against the MDE is the post-hoc power fallacy. The MDE is still returned, as a property of the design. New argument `clear_alpha=0.005`.
- `DStudy` has a new field `fixed_tasks`.

## 0.1.0 - 2026-10-17

Initial version: validation and balancing, bootstrap intervals, variance components and D-study, paired differences, MDE, rank stability, rerun agreement, pass@k, IQM, claim check, report and CLI.
