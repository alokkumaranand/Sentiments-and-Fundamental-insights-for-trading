"""Inference helpers: block bootstrap, Spearman correlation and Benjamini-Hochberg.

Design decision embodied here: **the resampling unit is the session, not the row.**

Forward returns at h>1 overlap (day d's h=5 return shares four days with day d+1's), daily
series are autocorrelated, and within one session the same news event is counted many times over.
An ordinary IID bootstrap would therefore report confidence intervals that are far too narrow —
a confident-looking answer to a question the data cannot answer that precisely. Blocks of
consecutive sessions (5 = one trading week) restore most of that dependence structure.

Second decision: **every p-value that reaches a report is adjusted for the batch it came from.**
Screening 25 tickers x 3 horizons is 75 tests, and at alpha=0.05 roughly four would look
significant by chance alone. BH controls the false discovery rate, which is the honest correction
for "we looked at a lot of things".
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (rho, p) for Spearman's rank correlation.

    Rank correlation because the relationships here are monotone at best and the distributions are
    far from normal (polarity is bounded, attention is heavily skewed). Uses scipy when available
    and a rank + Pearson fallback otherwise, so a missing optional dependency degrades the p-value
    but never the point estimate.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr

        result = spearmanr(x, y)
        return float(result.statistic), float(result.pvalue)
    except Exception:  # pragma: no cover - scipy is in requirements.txt
        return float(np.corrcoef(_rank(x), _rank(y))[0, 1]), float("nan")


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, the standard tie handling for a rank correlation."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # Average tied ranks so a tie does not depend on input order.
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def _block_index_groups(session_ids: np.ndarray, block_sessions: int) -> list[np.ndarray]:
    """Group row positions into consecutive-session blocks.

    Rows are grouped by session first, then sessions are chunked, so a block is a contiguous run
    of sessions regardless of how many rows each session has.
    """
    order = np.argsort(session_ids, kind="stable")
    sessions = session_ids[order]
    groups: list[np.ndarray] = []
    start = 0
    for end in range(1, len(sessions) + 1):
        if end == len(sessions) or sessions[end] != sessions[start]:
            groups.append(order[start:end])
            start = end
    return [np.concatenate(groups[i:i + block_sessions]) for i in range(0, len(groups), block_sessions)]


def block_bootstrap(
    x: np.ndarray,
    y: np.ndarray | None,
    statistic: Callable[[np.ndarray, np.ndarray | None], float],
    session_ids: np.ndarray,
    block_sessions: int = 5,
    resamples: int = 2000,
    seed: int = 20260912,
) -> list[float]:
    """Return the 95% percentile CI of `statistic` under a session-block bootstrap.

    `y=None` is allowed for single-series statistics (a mean, a bucket average). Returns
    [nan, nan] when there is not enough data to resample, which downstream code renders as
    "not estimable" rather than as a number.
    """
    x = np.asarray(x, dtype=float)
    y = None if y is None else np.asarray(y, dtype=float)
    if len(x) < 4:
        return [float("nan"), float("nan")]
    blocks = _block_index_groups(np.asarray(session_ids), max(1, block_sessions))
    if len(blocks) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=float)
    for i in range(resamples):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        index = np.concatenate([blocks[j] for j in chosen])
        values[i] = statistic(x[index], None if y is None else y[index])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [float("nan"), float("nan")]
    return [round(float(np.percentile(values, 2.5)), 6), round(float(np.percentile(values, 97.5)), 6)]


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Return BH-adjusted q-values for a batch, NaN-safe and order-preserving.

    The step-up procedure: sort ascending, scale each p by n/rank, then enforce monotonicity from
    the largest p down, so an adjusted value can never fall below a smaller one's.
    """
    values = np.asarray(pvalues, dtype=float)
    n = len(values)
    qvalues = np.full(n, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return [float("nan")] * n
    index = np.where(finite)[0]
    ordered = index[np.argsort(values[index], kind="stable")]
    m = len(ordered)
    scaled = values[ordered] * m / np.arange(1, m + 1)
    # Reverse cummin: q_i = min over j >= i of scaled_j.
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    qvalues[ordered] = np.clip(monotone, 0.0, 1.0)
    return [round(float(value), 6) if np.isfinite(value) else float("nan") for value in qvalues]


def mean_of(x: np.ndarray, y: np.ndarray | None = None) -> float:
    """Mean ignoring non-finite values. The default single-series statistic."""
    values = x[np.isfinite(x)]
    return float(values.mean()) if len(values) else float("nan")


def difference_of_means(x: np.ndarray, y: np.ndarray | None = None) -> float:
    """Mean of (x - y), used for paired comparisons such as bucket-mean minus unconditional mean."""
    if y is None:
        return mean_of(x)
    mask = np.isfinite(x) & np.isfinite(y)
    return float((x[mask] - y[mask]).mean()) if mask.any() else float("nan")


def robust_z(values: np.ndarray) -> np.ndarray:
    """Return the robust z-score of `values` (median/MAD), NaN when the scale is zero.

    1.4826 scales the MAD to a standard deviation for normal data, so a threshold of 3 reads like
    "three sigma" while remaining immune to the very spikes it is looking for.
    """
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad == 0:
        return np.full(len(values), np.nan)
    return (values - median) / (1.4826 * mad)
