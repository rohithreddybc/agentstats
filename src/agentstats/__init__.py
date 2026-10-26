"""agentstats: statistics for LLM agent benchmark results."""

from .bootstrap import CI, bootstrap_ci, iqm, iqm_ci
from .compare import (
    ClaimCheck,
    MDEResult,
    PairedDiff,
    claim_check,
    mde,
    mde_from_df,
    paired_diff,
)
from .data import balance, cell_means, system_scores, validate
from .passk import pass_at_k, pass_hat_k, system_pass_k
from .ranking import (
    RankStability,
    RerunAgreement,
    rank_stability,
    rerun_agreement,
    rerun_agreement_model,
    subsample_curve,
)
from .quantities import (
    PairZ,
    RunDisagreement,
    RunsNeeded,
    SECurve,
    fit_se_curve,
    pair_z,
    pair_z_table,
    reversal_probability,
    run_disagreement,
    runs_needed,
)
from .report import report
from .variance import (
    DStudy,
    VarianceComponents,
    VarianceComponentsSingle,
    d_study,
    trials_needed,
    variance_components,
    variance_components_array,
    variance_components_single,
)

__version__ = "0.2.0"

__all__ = [
    "CI",
    "ClaimCheck",
    "DStudy",
    "MDEResult",
    "PairZ",
    "PairedDiff",
    "RankStability",
    "RerunAgreement",
    "RunDisagreement",
    "RunsNeeded",
    "SECurve",
    "VarianceComponents",
    "VarianceComponentsSingle",
    "balance",
    "bootstrap_ci",
    "cell_means",
    "claim_check",
    "d_study",
    "fit_se_curve",
    "iqm",
    "iqm_ci",
    "mde",
    "mde_from_df",
    "pair_z",
    "pair_z_table",
    "paired_diff",
    "pass_at_k",
    "pass_hat_k",
    "rank_stability",
    "report",
    "reversal_probability",
    "run_disagreement",
    "runs_needed",
    "rerun_agreement",
    "rerun_agreement_model",
    "subsample_curve",
    "system_pass_k",
    "system_scores",
    "trials_needed",
    "validate",
    "variance_components",
    "variance_components_array",
    "variance_components_single",
]
