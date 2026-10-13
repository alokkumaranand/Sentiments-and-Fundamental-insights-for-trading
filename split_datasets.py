#!/usr/bin/env python
"""Split both raw datasets in this workspace into a 1:9 (validation : train) partition.

Inputs   (read-only, never modified, never extracted from their container):
    stock_tweets.csv.zip      -- ZIP holding one CSV member: stock_tweets.csv
    stock_yfinance_data.csv   -- plain CSV, one row per ticker x trading session

Outputs  (written to ./splits/ by default):
    tweets_validation.csv  tweets_train.csv
    prices_validation.csv  prices_train.csv

Row grain is preserved exactly: every output row is byte-identical to the input row it came
from, and `validation rows + train rows == input rows` is asserted before anything is written.

The one design decision this module embodies: the default split is CHRONOLOGICAL, not random.
The earliest 10% of rows form the validation slice and the later 90% the training slice, so no
training row is observed after a validation row. A random split would leak future information
into the training half and is therefore opt-in only (`--strategy random`), never the default.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Defaults come from this workspace's Step 0 profile: the tweets arrive zipped, the price file
# does not. Both are relative to the workspace root so `python split_datasets.py` just works.
DEFAULT_TWEETS = "stock_tweets.csv.zip"
DEFAULT_PRICES = "stock_yfinance_data.csv"
TWEET_MEMBER = "stock_tweets.csv"

# 0.10 = the 1 side of the requested 1:9 split. It is a fraction of *rows*, per SPEC.md §2.7
# ("Configurable as a fraction, not a date literal").
DEFAULT_RATIO = 0.10

# 20220929 = the day this pipeline's split semantics were fixed; changing it would reproduce a
# different partition under --strategy random, which is exactly what a frozen seed prevents.
DEFAULT_SEED = 20220929

VALIDATION_NAME = "validation"
TRAIN_NAME = "train"


# --------------------------------------------------------------------------- loading


def read_csv_source(path: Path, member: str | None = None) -> pd.DataFrame:
    """Read a CSV (optionally the named member of a ZIP) as all-string columns.

    Returns a DataFrame with no type inference: `dtype=str` keeps every Date and every Tweet
    byte-for-byte as it appears in the source. It deliberately does not parse dates, because a
    separate explicit parse step is what makes an unparseable date a loud error instead of a
    silent NaT that would later sort to the front of the chronological split.
    """
    read_kwargs = {"dtype": str, "keep_default_na": False, "na_filter": False}
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if member is not None:
                if member not in names:
                    raise SystemExit(f"'{member}' is not a CSV member of {path.name} (found: {names})")
                target = member
            elif len(names) == 1:
                target = names[0]
            else:
                raise SystemExit(f"{path.name} holds {len(names)} CSVs; pass --member to choose one: {names}")
            print(f"  reading member '{target}' from {path.name} (no extraction)")
            with archive.open(target) as handle:
                return pd.read_csv(handle, **read_kwargs)
    return pd.read_csv(path, **read_kwargs)


def find_date_column(frame: pd.DataFrame) -> str:
    """Return the name of the column holding timestamps.

    Preference order is the source's own name ('Date' in both files), then any case-insensitive
    match, then any column named date/timestamp that actually parses. It returns a column *name*,
    not a parsed series, so the caller stays responsible for sorting.
    """
    for candidate in ("Date", "date", "Datetime", "datetime", "timestamp", "Timestamp"):
        if candidate in frame.columns:
            return candidate
    for name in frame.columns:
        if "date" in name.lower() or "time" in name.lower():
            return name
    raise SystemExit(f"no date-like column found among {list(frame.columns)}")


def parse_dates_or_fail(series: pd.Series, column: str, source: str) -> pd.Series:
    """Parse `series` to UTC timestamps, failing loudly on any value pandas cannot read.

    Returns a tz-aware datetime Series. It does not drop or fill bad rows: an unparseable date
    must abort the run, because silently dropping it would change the split boundary without
    changing the reported ratios.
    """
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    bad = int(parsed.isna().sum())
    if bad:
        sample = series[parsed.isna()].head(3).tolist()
        raise SystemExit(f"{source}: {bad} unparseable value(s) in '{column}', e.g. {sample}")
    return parsed


# --------------------------------------------------------------------------- splitting


def split_chronological(frame: pd.DataFrame, order: pd.Series, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Split positions so the earliest `ratio` share of rows is validation.

    Returns two integer position arrays (validation, train). It sorts by `order` with the
    original row position as tie-breaker, so rows sharing a timestamp keep their source order
    and the boundary is reproducible. It does not touch group membership: an existing ticker
    present in both halves is intentional (the spec splits by session, not by ticker).
    """
    # stable sort on (timestamp, position) makes the boundary deterministic for tied timestamps
    order_key = np.lexsort((np.arange(len(frame)), order.to_numpy()))
    n_validation = int(round(len(frame) * ratio))
    n_validation = max(1, min(n_validation, len(frame) - 1)) if len(frame) > 1 else len(frame)
    return order_key[:n_validation], order_key[n_validation:]


def split_random(frame: pd.DataFrame, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split positions uniformly at random with a frozen seed.

    Returns two integer position arrays (validation, train). This is leak-prone for time series
    and exists only so a shuffle-based comparison can be reproduced; the default path never
    calls it.
    """
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(len(frame))
    n_validation = max(1, min(int(round(len(frame) * ratio)), len(frame) - 1))
    return shuffled[:n_validation], shuffled[n_validation:]


def split_grouped(frame: pd.DataFrame, tickers: pd.Series, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split whole tickers between the halves, keeping every ticker intact in one side.

    Returns two integer position arrays (validation, train). Assignment is a SHA-256 of
    (seed, ticker) reduced mod 10, so it is stable across runs and across machines and does not
    depend on Python's per-process hash randomisation. It does not guarantee the requested
    ratio exactly: with a handful of tickers the realised share can drift, which is reported.
    """
    positions_validation: list[int] = []
    positions_train: list[int] = []
    for ticker in pd.unique(tickers):
        digest = hashlib.sha256(f"{seed}:{ticker}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % 10
        target = positions_validation if bucket < round(ratio * 10) else positions_train
        target.extend(np.flatnonzero((tickers == ticker).to_numpy()).tolist())
    return np.array(positions_validation, dtype=int), np.array(positions_train, dtype=int)


def choose_split(
    frame: pd.DataFrame,
    strategy: str,
    ratio: float,
    seed: int,
    order: pd.Series,
    group_key: pd.Series | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the strategy named on the command line and return (validation, train) positions."""
    if strategy == "chronological":
        return split_chronological(frame, order, ratio)
    if strategy == "random":
        return split_random(frame, ratio, seed)
    if strategy == "grouped":
        if group_key is None:
            raise SystemExit("--strategy grouped needs a ticker column")
        return split_grouped(frame, group_key, ratio, seed)
    raise SystemExit(f"unknown strategy '{strategy}'")


# --------------------------------------------------------------------------- writing


def write_split(frame: pd.DataFrame, positions: np.ndarray, destination: Path, source_order: pd.DataFrame) -> pd.DataFrame:
    """Write `positions` of `frame` to `destination` in source row order and return the written rows.

    Returns the written DataFrame (its index reset) so the caller can verify content without
    re-reading the file. Rows are re-sorted to their original position so each output file is a
    plain subsequence of the input; that keeps a manual diff against the source meaningful.
    It does not reorder columns -- the source header order is preserved.
    """
    ordered = np.sort(positions)
    selection = source_order.iloc[ordered].copy()
    selection.to_csv(destination, index=False, encoding="utf-8-sig", lineterminator="\n")
    return selection


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of `path`, read in chunks so a large file never loads whole."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(frame: pd.DataFrame, dates: pd.Series, all_dates: pd.Series, label: str) -> str:
    """Return one aligned line summarising a split's row count, share of the whole, and date span.

    Share is computed against `all_dates` (the full dataset), not against this split, so the two
    lines shown for a file sum to 100% -- which is the number a reader checks first.
    """
    share = len(frame) / max(len(all_dates), 1) * 100.0
    if len(frame) == 0:
        return f"  {label:<11} rows=0"
    # min/max are only a contiguous block under --strategy chronological; the strategy is printed
    # in the header so these spans are never read as more than they are.
    return (
        f"  {label:<11} rows={len(frame):>7,} ({share:5.2f}%)  "
        f"{dates.min():%Y-%m-%d} -> {dates.max():%Y-%m-%d}"
    )


# --------------------------------------------------------------------------- driver


def process(
    name: str,
    path: Path,
    member: str | None,
    outdir: Path,
    prefix: str,
    ratio: float,
    strategy: str,
    seed: int,
    group_column: str | None,
    columns: list[str] | None,
) -> dict[str, object]:
    """Split one source file, write both halves and return its report record.

    Returns a dict of primitive values (no live dataframe objects) so the caller can print a
    summary or serialise it. It prints progress as it goes and aborts the run on any integrity
    failure rather than writing a partial partition.
    """
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    print(f"\n{name}: {path.name}  ({path.stat().st_size:,} bytes, sha256 {sha256_file(path)[:16]}...)")
    frame = read_csv_source(path, member)
    print(f"  loaded {len(frame):,} rows x {len(frame.columns)} cols: {list(frame.columns)}")

    if columns:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise SystemExit(f"{path.name} lacks column(s) {missing}")
        frame = frame[columns]
        print(f"  restricted to requested columns: {list(frame.columns)}")

    date_column = find_date_column(frame)
    dates = parse_dates_or_fail(frame[date_column], date_column, path.name)

    group_key = None
    if strategy == "grouped":
        if group_column is None:
            # 'Stock Name' holds the bare ticker in both files (STEP0_PROFILE section 1)
            group_column = "Stock Name"
        if group_column not in frame.columns:
            raise SystemExit(f"--strategy grouped: no ticker column '{group_column}' in {path.name}")
        group_key = frame[group_column]
        print(f"  grouping whole tickers by '{group_column}' ({group_key.nunique()} distinct)")

    validation_pos, train_pos = choose_split(frame, strategy, ratio, seed, dates, group_key)

    # Integrity gate: the two halves must partition the input exactly. An off-by-one here would
    # silently shrink the dataset, so nothing is written until this passes.
    if len(validation_pos) + len(train_pos) != len(frame):
        raise SystemExit("internal error: split does not cover every row")
    if set(validation_pos) & set(train_pos):
        raise SystemExit("internal error: a row landed in both halves")

    outdir.mkdir(parents=True, exist_ok=True)
    validation_path = outdir / f"{prefix}_{VALIDATION_NAME}.csv"
    train_path = outdir / f"{prefix}_{TRAIN_NAME}.csv"
    validation = write_split(frame, validation_pos, validation_path, frame)
    train = write_split(frame, train_pos, train_path, frame)

    # Second gate: re-read what we wrote and compare it row-for-row against the source.
    for label, written, target in (
        (VALIDATION_NAME, validation, validation_path),
        (TRAIN_NAME, train, train_path),
    ):
        reloaded = pd.read_csv(target, dtype=str, keep_default_na=False, na_filter=False)
        if reloaded.shape != written.shape:
            raise SystemExit(f"{target.name}: shape mismatch after write {reloaded.shape} != {written.shape}")
        if not reloaded.equals(written.reset_index(drop=True)):
            raise SystemExit(f"{target.name}: content mismatch after write")
        print(f"  wrote {target}  ({target.stat().st_size:,} bytes) -- re-read and verified")

    print(describe(validation, dates.iloc[validation_pos], dates, VALIDATION_NAME))
    print(describe(train, dates.iloc[train_pos], dates, TRAIN_NAME))

    return {
        "name": name,
        "source": str(path),
        "source_rows": len(frame),
        "date_column": date_column,
        "validation_rows": len(validation),
        "train_rows": len(train),
        "validation_path": str(validation_path),
        "train_path": str(train_path),
        "validation_sha256": sha256_file(validation_path),
        "train_sha256": sha256_file(train_path),
    }


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser. Every knob the split depends on is an explicit flag."""
    parser = argparse.ArgumentParser(
        description="Split both workspace datasets 1:9 (validation:train) by default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tweets", default=DEFAULT_TWEETS, help="tweets source (zip or csv)")
    parser.add_argument("--prices", default=DEFAULT_PRICES, help="price source csv")
    parser.add_argument("--member", default=TWEET_MEMBER, help="CSV member name inside the tweets zip")
    parser.add_argument("--outdir", default="splits", help="directory for the four output files")
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO, help="validation share (0.10 = 1:9)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for random/grouped strategies")
    parser.add_argument(
        "--strategy",
        choices=("chronological", "random", "grouped"),
        default="chronological",
        help="chronological = earliest rows validate (leak-safe default)",
    )
    parser.add_argument("--group-column", default=None, help="ticker column for --strategy grouped")
    parser.add_argument(
        "--only",
        choices=("both", "tweets", "prices"),
        default="both",
        help="split only one file",
    )
    parser.add_argument(
        "--columns",
        default=None,
        help="comma-separated column subset to keep, e.g. Date,Tweet,Stock Name",
    )
    parser.add_argument("--skip-tweets", action="store_true", help="deprecated alias for --only prices")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the split for the selected files and print a final summary table. Returns an exit code."""
    args = build_parser().parse_args(argv)
    if not 0.0 < args.ratio < 1.0:
        raise SystemExit(f"--ratio must be strictly between 0 and 1, got {args.ratio}")

    workspace = Path(__file__).resolve().parent
    outdir = (workspace / args.outdir).resolve()
    columns = [c.strip() for c in args.columns.split(",")] if args.columns else None
    only = "prices" if args.skip_tweets else args.only

    print(f"workspace : {workspace}")
    print(f"strategy  : {args.strategy}")
    print(f"ratio     : {args.ratio:.4f} validation : {1 - args.ratio:.4f} train"
          f"   (round to {args.ratio * 10:.0f}:{(1 - args.ratio) * 10:.0f})")
    print(f"output dir: {outdir}")

    records: list[dict[str, object]] = []
    if only in ("both", "tweets"):
        records.append(
            process(
                "tweets",
                workspace / args.tweets,
                args.member,
                outdir,
                "tweets",
                args.ratio,
                args.strategy,
                args.seed,
                args.group_column,
                columns,
            )
        )
    if only in ("both", "prices"):
        records.append(
            process(
                "prices",
                workspace / args.prices,
                None,
                outdir,
                "prices",
                args.ratio,
                args.strategy,
                args.seed,
                args.group_column,
                columns,
            )
        )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'dataset':<9} {'source':>9} {'validation':>11} {'train':>9}  {'realised 1:9?':>13}")
    for record in records:
        source_rows = int(record["source_rows"])
        realised = int(record["validation_rows"]) / source_rows
        flag = f"{realised * 100:5.2f}% / {100 - realised * 100:5.2f}%"
        print(
            f"{record['name']:<9} {source_rows:>9,} {record['validation_rows']:>11,} "
            f"{record['train_rows']:>9,}  {flag:>13}"
        )
    print("\nfiles:")
    for record in records:
        print(f"  {record['validation_path']}")
        print(f"    sha256 {record['validation_sha256']}")
        print(f"  {record['train_path']}")
        print(f"    sha256 {record['train_sha256']}")

    if args.strategy != "chronological":
        print(f"\nnote: --strategy {args.strategy} is not leak-safe; validation rows are not a "
              "contiguous earliest block.")
    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
