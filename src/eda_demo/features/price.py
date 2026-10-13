"""Price features: returns, realized volatility and a past-only volume z-score.

Design decision embodied here: **two kinds of return, never mixed.** `ret_contemp` (close d-1 ->
close d) is descriptive co-movement only; `fwd_ret_h` (close d -> close d+h) is the only return
used in predictive-style analysis. The names say which is which so a caption cannot be written
that quietly turns a same-day co-movement into a forecast (SPEC.md 4.2, 4.7).

The last `h` sessions of a ticker have a null `fwd_ret_h`, not a wrapped value: there is no
future price in the window to measure, and inventing one would fabricate evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# 20 sessions = one trading month, the conventional short baseline for a z-score.
_VOLUME_BASELINE = 20


def price_features(prices: pd.DataFrame, horizons: list[int], vol_windows: list[int]) -> pd.DataFrame:
    """Return the price feature frame, one row per ticker-session.

    `return_basis` is `adj_close` throughout: it is verified split-adjusted and differs from
    `close` only by dividends, so it is the right basis for a return and needs no split handling.
    """
    frame = prices.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    per_ticker = frame.groupby("ticker", observed=True, sort=False)

    frame["ret_contemp"] = per_ticker["adj_close"].pct_change()
    frame["abs_ret"] = frame["ret_contemp"].abs()
    for horizon in horizons:
        # shift(-h) looks forward in time; that is the point, and it is named so it cannot be
        # mistaken for a feature available at decision time.
        frame[f"fwd_ret_{horizon}"] = per_ticker["adj_close"].shift(-horizon) / frame["adj_close"] - 1.0
    for window in vol_windows:
        # Realized volatility uses returns through day d (past and present day only).
        frame[f"rv_{window}"] = per_ticker["ret_contemp"].transform(
            lambda series: series.rolling(window, min_periods=max(2, window // 2)).std()
        )

    # Volume z-score against a trailing 20-session baseline, shifted so today's volume is never
    # part of its own baseline. log1p first: volume is heavily right-skewed and one promo blast
    # would otherwise dominate the mean.
    frame["log_volume"] = np.log1p(frame["volume"].clip(lower=0))
    per_ticker = frame.groupby("ticker", observed=True, sort=False)
    baseline = per_ticker["log_volume"].transform(lambda series: series.shift(1))
    mean = baseline.groupby(frame["ticker"], observed=True, sort=False).transform(
        lambda series: series.rolling(_VOLUME_BASELINE, min_periods=5).mean()
    )
    std = baseline.groupby(frame["ticker"], observed=True, sort=False).transform(
        lambda series: series.rolling(_VOLUME_BASELINE, min_periods=5).std()
    )
    frame["volume_z"] = (frame["log_volume"] - mean) / std.replace(0.0, np.nan)
    return frame


def price_report(features: pd.DataFrame, horizons: list[int]) -> dict[str, Any]:
    """Summarise the price frame for the feature book and the terminal overview."""
    last = features.groupby("ticker", observed=True).tail(1)
    return {
        "n_units": int(len(features)),
        "sessions_per_ticker": int(features.groupby("ticker", observed=True).size().max()),
        "mean_abs_ret": round(float(features["abs_ret"].mean()), 6),
        "null_fwd_returns_at_tail": {f"h={h}": int(last[f"fwd_ret_{h}"].isna().sum()) for h in horizons},
        "volume_z_available": int(features["volume_z"].notna().sum()),
    }
