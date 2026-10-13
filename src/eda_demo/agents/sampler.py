"""The capped, stratified sample the inductor is allowed to read.

Design decision embodied here: **the sample is chosen by hash, not by the model and not by
`sample()`.** A deterministic hash order means the same corpus always yields the same sample, so
a proposal can be re-derived and audited; a random sample would make the whole induction stage
irreproducible for the sake of nothing.

Second decision: **the sample and the validation slice are separated by session parity.** A term
must be checked on text the inductor never saw, and parity gives both halves of the corpus the
whole window — so a word cannot pass or fail merely because it happens to appear only late in
the year, which is exactly what a chronological cut would have caused.

Third decision: **the sample is stratified, with caps on the awkward strata.** Promo blasts,
replies and cashtag-dense tweets are each a large minority of this corpus and are linguistically
unlike ordinary commentary; a uniform sample would mostly show promo text and the proposal would
be about advertising rather than about sentiment.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class InductionSample:
    """The prompt rows, the disjoint validation rows, and a report of how they were drawn."""

    sample: pd.DataFrame
    validation: pd.DataFrame
    report: dict[str, Any]


def _row_hash(row: pd.Series) -> str:
    """Stable per-row hash. Takes no random seed, so the sample cannot drift between runs."""
    key = f"{row['ticker']}|{row['session_date']}|{row['text_key']}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _stratum(row: pd.Series) -> str:
    """Label a tweet with the stratum it belongs to, most specific first."""
    if row["promo_flag"]:
        return "promo"
    if row["reply_flag"]:
        return "reply"
    if len(row["cashtags"]) >= 2:
        return "cashtag_dense"
    return "ordinary"


def sample_for_induction(
    tweets: pd.DataFrame,
    per_ticker: int,
    max_total: int,
    validation_parity: int,
    strata_caps: dict[str, float],
) -> InductionSample:
    """Return (sample, validation) frames plus a report of the draw.

    Rows are hashed, sorted within each ticker and taken stratum-first up to the caps, then filled
    from the remainder. Everything is deterministic: same corpus in, same sample out.
    """
    frame = tweets[tweets["session_date"].notna()].copy()
    # Column names here avoid a leading underscore on purpose: `itertuples` renames identifiers it
    # considers invalid, and `row._hash` silently becomes `row._1`.
    frame["sample_hash"] = frame.apply(_row_hash, axis=1)
    frame["stratum"] = frame.apply(_stratum, axis=1)

    sessions = sorted(frame["session_date"].unique())
    parity = {session: index % 2 for index, session in enumerate(sessions)}
    frame["parity"] = frame["session_date"].map(parity)
    sample_pool = frame[frame["parity"] != validation_parity]
    validation = frame[frame["parity"] == validation_parity]

    picked: list[pd.DataFrame] = []
    for _, group in sample_pool.groupby("ticker", observed=True):
        ordered = group.sort_values("sample_hash")
        quotas = {name: int(per_ticker * cap) for name, cap in strata_caps.items()}
        chosen = pd.DataFrame()
        for name, quota in quotas.items():
            if quota <= 0:
                continue
            chosen = pd.concat([chosen, ordered[ordered["stratum"] == name].head(quota)])
        remaining = per_ticker - len(chosen)
        if remaining > 0:
            rest = ordered[~ordered.index.isin(chosen.index)]
            chosen = pd.concat([chosen, rest.head(remaining)])
        picked.append(chosen)

    sample = pd.concat(picked) if picked else sample_pool.head(0)
    sample = sample.sort_values("sample_hash")
    if len(sample) > max_total:
        # Truncating by hash keeps the draw unbiased across tickers instead of favouring the
        # first tickers alphabetically.
        sample = sample.head(max_total)

    report = {
        "sample_rows": int(len(sample)),
        "validation_rows": int(len(validation)),
        "sample_sessions": int(sample["session_date"].nunique()),
        "validation_sessions": int(validation["session_date"].nunique()),        "sessions_total": int(len(sessions)),
        "validation_parity": int(validation_parity),
        "per_ticker": int(per_ticker),
        "max_total": int(max_total),
        "strata_caps": dict(strata_caps),
        "sample_strata": {str(k): int(v) for k, v in sample["stratum"].value_counts().items()},
        "tickers_covered": int(sample["ticker"].nunique()),
        "method": "deterministic sha256(ticker|session|text) order, strata-capped, session-disjoint",
    }
    keep = ["ticker", "session_date", "text_clean", "stratum", "sample_hash"]
    return InductionSample(sample=sample[keep], validation=validation, report=report)


def format_sample(sample: pd.DataFrame, max_chars: int = 220) -> str:
    """Render the sample as compact lines: `[stratum] TICKER text`.

    Truncating each tweet keeps the prompt bounded; 220 characters covers this corpus's p99 and
    the truncated tail is promo boilerplate rather than sentiment.
    """
    lines = []
    for row in sample.itertuples():
        text = str(row.text_clean)[:max_chars].replace("\n", " ")
        lines.append(f"[{row.stratum}] {row.ticker}: {text}")
    return "\n".join(lines)
