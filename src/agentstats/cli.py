"""Command line interface: ``agentstats report path.csv|path.parquet [--benchmark B] [--seed S]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .report import report

_STR_COLS = {c: str for c in ("benchmark", "task_id", "system_id", "model", "harness", "source")}


def _read(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=_STR_COLS)
    if suffix in (".parquet", ".pq"):
        try:
            return pd.read_parquet(path)
        except ImportError as e:  # pyarrow/fastparquet are optional
            raise ValueError(
                "reading parquet needs an optional engine (pip install pyarrow); "
                f"original error: {e}"
            ) from e
    raise ValueError(f"unsupported file type {suffix!r}; use .csv or .parquet")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agentstats", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("report", help="print a markdown reporting card")
    rp.add_argument("path", type=Path)
    rp.add_argument("--benchmark", default=None, help="benchmark to report (required if several)")
    rp.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    rp.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples (default 2000)")
    args = parser.parse_args(argv)
    try:
        df = _read(args.path)
        if args.benchmark is not None:
            if "benchmark" not in df.columns:
                raise ValueError("file has no 'benchmark' column")
            df = df[df["benchmark"] == args.benchmark]
            if df.empty:
                raise ValueError(f"no rows for benchmark {args.benchmark!r}")
        print(report(df, n_boot=args.n_boot, rng=args.seed))
    except (ValueError, OSError) as e:
        print(f"agentstats: error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
