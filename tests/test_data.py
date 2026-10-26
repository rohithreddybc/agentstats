import numpy as np
import pandas as pd
import pytest

import agentstats as ast
from _sim import df_from_array


def base_df():
    rng = np.random.default_rng(0)
    return df_from_array(rng.random((3, 4, 2)))


def test_validate_ok_returns_frame():
    df = base_df()
    out = ast.validate(df)
    assert len(out) == len(df)
    assert out["run_id"].dtype == np.int64 and out["score"].dtype == np.float64


def test_validate_missing_columns():
    with pytest.raises(ValueError, match="missing required column"):
        ast.validate(base_df().drop(columns=["score"]))


def test_validate_out_of_range_scores():
    df = base_df()
    df.loc[0, "score"] = 1.2
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ast.validate(df)
    df.loc[0, "score"] = -0.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ast.validate(df)


def test_validate_nan_score():
    df = base_df()
    df.loc[3, "score"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ast.validate(df)


def test_validate_duplicate_keys():
    df = base_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        ast.validate(df)


def test_validate_bad_dtypes():
    df = base_df()
    df["task_id"] = np.arange(len(df)) % 4  # int ids are not str
    with pytest.raises(ValueError, match="task_id"):
        ast.validate(df)
    df = base_df()
    df["run_id"] = df["run_id"].astype(float)
    with pytest.raises(ValueError, match="run_id"):
        ast.validate(df)
    with pytest.raises(ValueError):
        ast.validate(base_df().iloc[0:0])
    with pytest.raises(ValueError):
        ast.validate([1, 2, 3])


def test_single_benchmark_enforced():
    a = df_from_array(np.random.default_rng(0).random((2, 5, 2)), benchmark="x")
    b = df_from_array(np.random.default_rng(1).random((2, 5, 2)), benchmark="y")
    with pytest.raises(ValueError, match="benchmarks"):
        ast.bootstrap_ci(pd.concat([a, b]), "s000", n_boot=20, rng=0)


def _ragged_df():
    rows = []
    # (system, task) -> number of runs
    runs = {
        ("A", "t0"): 3, ("A", "t1"): 3, ("A", "t2"): 2, ("A", "t3"): 3,
        ("B", "t0"): 4, ("B", "t1"): 3, ("B", "t2"): 3,  # B has no t3
    }
    for (s, t), n in runs.items():
        for r in range(n):
            rows.append(("b", t, s, r, 0.5))
    return pd.DataFrame(rows, columns=["benchmark", "task_id", "system_id", "run_id", "score"])


def test_balance_drops_right_tasks_and_reports_counts():
    out, info = ast.balance(_ragged_df(), min_runs=3)
    assert sorted(out["task_id"].unique()) == ["t0", "t1"]
    assert info["dropped_tasks"] == ["t2", "t3"]
    assert info["n_tasks_total"] == 4 and info["n_tasks_kept"] == 2 and info["n_tasks_dropped"] == 2
    sizes = out.groupby(["system_id", "task_id"]).size()
    assert (sizes == 3).all() and len(sizes) == 4
    # B/t0 had 4 runs: first 3 run_ids kept, 1 attempt truncated
    assert info["rows_truncated"] == 1
    assert sorted(out.loc[(out.system_id == "B") & (out.task_id == "t0"), "run_id"]) == [0, 1, 2]
    # rows: A t2 (2) + A t3 (3) dropped with their tasks, B t2 (3) too
    assert info["rows_dropped_with_tasks"] == 2 + 3 + 3
    assert info["rows_in"] == len(_ragged_df())
    assert info["rows_out"] == len(out) == 12


def test_balance_system_subset_and_failure():
    out, info = ast.balance(_ragged_df(), min_runs=3, systems=["A"])
    assert sorted(out["task_id"].unique()) == ["t0", "t1", "t3"]
    assert set(out["system_id"]) == {"A"}
    assert info["dropped_tasks"] == ["t2"]
    with pytest.raises(ValueError, match="no task"):
        ast.balance(_ragged_df(), min_runs=5)
    with pytest.raises(ValueError, match="unknown"):
        ast.balance(_ragged_df(), min_runs=1, systems=["Z"])


def test_cell_means_and_system_scores():
    df = _ragged_df()
    df.loc[(df.system_id == "A") & (df.task_id == "t0"), "score"] = [0.0, 0.5, 1.0]
    cm = ast.cell_means(df)
    row = cm[(cm.system_id == "A") & (cm.task_id == "t0")].iloc[0]
    assert row["mean"] == pytest.approx(0.5) and row["n_runs"] == 3
    ss = ast.system_scores(df).set_index("system_id")
    assert ss.loc["A", "n_tasks"] == 4 and ss.loc["B", "n_tasks"] == 3
    assert ss.loc["A", "n_runs_min"] == 2 and ss.loc["A", "n_runs_max"] == 3
    assert ss.loc["B", "n_runs_max"] == 4
