import subprocess
import sys

import numpy as np
import pytest

import agentstats as ast
from agentstats.cli import main
from _sim import bernoulli_tasks_df


def tiny_csv(tmp_path, with_second_benchmark=False):
    rng = np.random.default_rng(0)
    df = bernoulli_tasks_df(rng, 3, 8, 3, skills=[0.0, 0.5, 1.0], sd_task=0.6, benchmark="bench-a")
    if with_second_benchmark:
        import pandas as pd

        df = pd.concat([df, df.assign(benchmark="bench-b")], ignore_index=True)
    p = tmp_path / "tiny.csv"
    df.to_csv(p, index=False)
    return p


def test_cli_smoke(tmp_path, capsys):
    rc = main(["report", str(tiny_csv(tmp_path)), "--seed", "1", "--n-boot", "200"])
    out = capsys.readouterr().out
    assert rc == 0
    for section in ("# Benchmark reporting card: bench-a", "## Scores", "## Variance components",
                    "## D-study", "## Rank stability", "## Rerun agreement", "## Minimum detectable", "## What authors should report"):
        assert section in out
    assert "s000" in out and "s002" in out


def test_cli_is_reproducible_and_benchmark_filter(tmp_path, capsys):
    p = tiny_csv(tmp_path, with_second_benchmark=True)
    assert main(["report", str(p), "--benchmark", "bench-b", "--seed", "3", "--n-boot", "150"]) == 0
    a = capsys.readouterr().out
    assert main(["report", str(p), "--benchmark", "bench-b", "--seed", "3", "--n-boot", "150"]) == 0
    assert capsys.readouterr().out == a and "bench-b" in a
    assert main(["report", str(p)]) == 2  # two benchmarks, none chosen
    err = capsys.readouterr().err
    assert "benchmarks" in err
    assert main(["report", str(p), "--benchmark", "nope"]) == 2
    assert main(["report", str(tmp_path / "missing.csv")]) == 2
    other = tmp_path / "x.txt"
    other.write_text("a")
    assert main(["report", str(other)]) == 2


def test_cli_as_module(tmp_path):
    p = tiny_csv(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "agentstats", "report", str(p), "--n-boot", "100"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Scores" in proc.stdout


def test_report_handles_single_run_and_single_system():
    rng = np.random.default_rng(1)
    one_run = bernoulli_tasks_df(rng, 3, 20, 1, skills=[0, 0.3, 0.6])
    txt = ast.report(one_run, n_boot=100, rng=0)
    assert "Only one run per cell" in txt and "Not available" in txt
    single = bernoulli_tasks_df(rng, 1, 20, 4, skills=[0.0])
    txt = ast.report(single, n_boot=100, rng=0)
    assert "ICC(1)" in txt and "single system" in txt


def test_report_notes_excluded_tasks():
    rng = np.random.default_rng(2)
    df = bernoulli_tasks_df(rng, 3, 20, 4, skills=[0, 0.3, 0.6])
    df = df[~((df.system_id == "s001") & (df.task_id == "t00003"))]
    txt = ast.report(df, n_boot=100, rng=0)
    assert "1 task(s) not evaluated by every system were excluded" in txt


def test_report_uses_rerun_agreement_and_new_defaults():
    rng = np.random.default_rng(4)
    df = bernoulli_tasks_df(rng, 4, 40, 4, skills=[0, 0.2, 0.4, 0.6])
    txt = ast.report(df, n_boot=100, rng=0)
    assert "split-half, same tasks" in txt and "model, full run count" in txt
    assert "biased upward" in txt
