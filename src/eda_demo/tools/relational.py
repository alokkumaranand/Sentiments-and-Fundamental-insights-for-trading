"""Relational tools: correlation, conditional forward returns, correlation matrix, spikes.

These are the tools that produce a *test*, so they are also the tools whose results carry a
p-value, a bootstrap CI and an `n_units`. Three decisions are visible in every one of them:

- **Sessions are the resampling unit**, so a CI reflects the fact that one news event is counted
  many times (see `stats/inference.py`).
- **The direction of time is explicit.** A same-day relationship is described as co-movement; only
  `fwd_ret_h` is allowed in a predictive-style statement, and its column name says so.
- **Pooling standardises within ticker first**, so a pooled number is about ticker-days rather than
  about which ticker posted the most.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from ..stats.inference import block_bootstrap, difference_of_means, mean_of, robust_z, spearman
from .common import (
    bar_chart,
    line_chart,
    metric_column,
    numeric,
    values_of,
    within_ticker_z,
)
from .registry import ToolContext, ToolError, ToolParams
from .schema import ChartSpec


class CorrelateParams(ToolParams):
    """Parameters of `correlate`."""

    x: str = Field(description="panel column used as the explanatory variable")
    y: str = Field(description="panel column used as the response variable")
    lag: int = Field(default=0, ge=-5, le=5, description="sessions to shift x forward; lag=1 tests x leading y")
    scope: str = Field(default="pooled", description="pooled | pooled_ex_TSLA | ticker")
    ticker: str | None = None
    period: str = Field(default="discovery")


class BucketParams(ToolParams):
    """Parameters of `bucket_forward_returns`."""

    feature: str = Field(description="panel column to bucket, e.g. sent_net or attn_z")
    horizon: int = Field(default=1, ge=1, le=5, description="forward-return horizon in sessions")
    buckets: int = Field(default=3, ge=2, le=5)
    scope: str = Field(default="pooled", description="pooled | pooled_ex_TSLA | ticker")
    ticker: str | None = None
    period: str = Field(default="discovery")


class CorrMatrixParams(ToolParams):
    """Parameters of `corr_matrix`."""

    metrics: list[str] = Field(min_length=2, max_length=8, description="2-8 panel columns")
    level: str = Field(default="pooled", description="pooled (within-ticker standardised) | per_ticker (mean pairwise)")
    period: str = Field(default="discovery")


class SpikeParams(ToolParams):
    """Parameters of `spike_days`."""

    metric: str = Field(default="n_texts_own", description="panel column to look for spikes in")
    z_threshold: float = Field(default=3.0, ge=1.5, le=6.0)
    scope: str = Field(default="ticker", description="ticker | pooled | pooled_ex_TSLA")
    ticker: str | None = None
    period: str = Field(default="discovery")


class TickerPairParams(ToolParams):
    """Parameters of the two dual-axis tools."""

    ticker: str
    period: str = Field(default="discovery")


def register(registry: Any) -> None:
    """Register the relational tools on `registry`."""

    @registry.add("correlate", CorrelateParams,
                  "Spearman correlation between two panel columns with a block-bootstrap CI.",
                  chart_kind="scatter")
    def correlate(context: ToolContext, params: CorrelateParams) -> dict[str, Any]:
        x_col = metric_column(context, params.x)
        y_col = metric_column(context, params.y)
        selection = context.select(period=params.period, scope=params.scope, ticker=params.ticker)
        frame = selection.frame.sort_values(["ticker", "session_idx"]).copy()
        if params.lag:
            # Shift x forward within its own ticker: lag=1 compares x on day d-1 with y on day d,
            # which is the only arrangement that can support a "leads" reading.
            frame[x_col] = frame.groupby("ticker", observed=True)[x_col].shift(params.lag)
        frame = frame[frame[x_col].notna() & frame[y_col].notna()]
        n_dropped = selection.n_dropped + (selection.n_units - len(frame))

        if len(frame) < 8:
            return {
                "n_obs": 0, "n_units": int(len(frame)),
                "stats": {"x": x_col, "y": y_col, "lag": params.lag, "note": "too few usable units to correlate"},
                "caveats": selection.caveats + ["correlation needs at least 8 usable ticker-sessions"],
            }

        if params.scope == "ticker":
            x_values = values_of(frame, x_col).to_numpy()
            y_values = values_of(frame, y_col).to_numpy()
            standardised = False
        else:
            x_values = within_ticker_z(frame, x_col).to_numpy()
            y_values = within_ticker_z(frame, y_col).to_numpy()
            standardised = True
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_values, y_values = x_values[mask], y_values[mask]
        sessions = frame["session_idx"].to_numpy()[mask]

        rho, p_value = spearman(x_values, y_values)
        ci = block_bootstrap(
            x_values, y_values,
            lambda a, b: spearman(a, b)[0],
            sessions,
            block_sessions=int(context.settings.experiments.get("bootstrap", {}).get("block_sessions", 5)),
            resamples=int(context.settings.experiments.get("bootstrap", {}).get("resamples", 2000)),
            seed=int(context.settings.experiments.get("bootstrap", {}).get("seed", 20260912)),
        )
        stats = {
            "x": x_col,
            "y": y_col,
            "lag_sessions": params.lag,
            "spearman_rho": numeric(rho),
            "p_value": numeric(p_value),
            "ci_95": [numeric(ci[0]), numeric(ci[1])],
            "n_obs": int(len(x_values)),
            "n_units": int(len(x_values)),
            "standardised_within_ticker": standardised,
            "direction": _direction(x_col, y_col, params.lag),
        }
        step = max(1, len(x_values) // 300)
        chart = ChartSpec(
            kind="scatter",
            title=f"{y_col} vs {x_col} (lag {params.lag})",
            x={"label": x_col, "values": [numeric(v) for v in x_values[::step]]},
            series=[{"label": y_col, "values": [numeric(v) for v in y_values[::step]]}],
            caption=f"{y_col} vs {x_col}; period={params.period}; n_units={len(x_values)}; rho={numeric(rho)}",
        )
        return {
            "n_obs": int(len(x_values)),
            "n_units": int(len(x_values)),
            "n_dropped": int(max(0, n_dropped)),
            "stats": stats,
            "chart": chart,
            "caveats": [
                "rank correlation on ticker-sessions: a co-movement, never a cause",
                stats["direction"],
                "CI is a session-block bootstrap because daily series are autocorrelated",
            ] + selection.caveats,
        }

    @registry.add("bucket_forward_returns", BucketParams,
                  "Mean forward return per feature bucket versus the unconditional mean.",
                  chart_kind="bar_error")
    def bucket_forward_returns(context: ToolContext, params: BucketParams) -> dict[str, Any]:
        feature = metric_column(context, params.feature)
        target = f"fwd_ret_{params.horizon}"
        if target not in context.panel.columns:
            raise ToolError(f"horizon {params.horizon} is not configured; available: "
                            f"{[c for c in context.panel.columns if c.startswith('fwd_ret_')]}")
        selection = context.select(period=params.period, scope=params.scope, ticker=params.ticker,
                                   require=(target,))
        frame = selection.frame.copy()
        if len(frame) < params.buckets * 4:
            return {
                "n_obs": int(len(frame)), "n_units": int(len(frame)),
                "stats": {"feature": feature, "horizon": params.horizon,
                          "note": "too few usable units for this many buckets"},
                "caveats": selection.caveats + ["needs at least 4 units per bucket"],
            }

        # Buckets are cut on the within-ticker standardised feature so a pooled tercile means
        # "unusually high for this ticker", not "this ticker".
        if params.scope == "ticker":
            scores = values_of(frame, feature)
        else:
            scores = within_ticker_z(frame, feature)
        keep = scores.notna() & values_of(frame, target).notna()
        frame = frame[keep].copy()
        scores = scores[keep]

        edges = np.unique(np.quantile(scores.to_numpy(), np.linspace(0, 1, params.buckets + 1)))
        if len(edges) < params.buckets + 1:
            return {
                "n_obs": int(len(frame)), "n_units": int(len(frame)),
                "stats": {"feature": feature, "horizon": params.horizon,
                          "note": "the feature has too few distinct values to cut distinct buckets"},
                "caveats": selection.caveats,
            }
        labels = [f"q{i + 1}" for i in range(params.buckets)]
        assignments = pd.cut(scores, bins=edges, labels=labels, include_lowest=True)

        returns = values_of(frame, target)
        sessions = frame["session_idx"].to_numpy()
        bootstrap_cfg = context.settings.experiments.get("bootstrap", {})
        block = int(bootstrap_cfg.get("block_sessions", 5))
        resamples = int(bootstrap_cfg.get("resamples", 2000))
        seed = int(bootstrap_cfg.get("seed", 20260912))

        bucket_rows: list[dict[str, Any]] = []
        means: list[float | None] = []
        errors: list[float | None] = []
        for label in labels:
            mask = (assignments == label).to_numpy()
            values = returns.to_numpy()[mask]
            if len(values) == 0:
                bucket_rows.append({"bucket": label, "n_units": 0, "mean": None, "ci_95": [None, None]})
                means.append(None)
                errors.append(None)
                continue
            ci = block_bootstrap(values, None, mean_of, sessions[mask], block, resamples, seed)
            bucket_rows.append({
                "bucket": label,
                "n_units": int(len(values)),
                "mean": numeric(values.mean()),
                "ci_95": [numeric(ci[0]), numeric(ci[1])],
            })
            means.append(numeric(values.mean()))
            errors.append(numeric((ci[1] - ci[0]) / 2) if ci[0] == ci[0] else None)

        unconditional = float(returns.mean())
        top = returns.to_numpy()[(assignments == labels[-1]).to_numpy()]
        bottom = returns.to_numpy()[(assignments == labels[0]).to_numpy()]
        spread_ci = [None, None]
        spread_p = None
        spread = None
        if len(top) and len(bottom):
            spread = float(top.mean() - bottom.mean())
            spread_ci = block_bootstrap(
                returns.to_numpy(),
                np.where((assignments == labels[-1]).to_numpy(), 1.0,
                         np.where((assignments == labels[0]).to_numpy(), -1.0, 0.0)),
                difference_of_means,
                sessions,
                block,
                resamples,
                seed,
            )
            spread_p = _bootstrap_p_value(
                returns.to_numpy(),
                np.where((assignments == labels[-1]).to_numpy(), 1.0,
                         np.where((assignments == labels[0]).to_numpy(), -1.0, 0.0)),
                sessions, block, resamples, seed,
            )

        stats = {
            "feature": feature,
            "horizon_sessions": params.horizon,
            "target": target,
            "unconditional_mean": numeric(unconditional),
            "buckets": bucket_rows,
            "spread_top_minus_bottom": numeric(spread),
            "spread_ci_95": [numeric(spread_ci[0]), numeric(spread_ci[1])],
            "p_value": numeric(spread_p),
            "n_units": int(len(frame)),
        }
        chart = bar_chart(
            f"{feature} bucket -> {target}",
            "bucket",
            labels,
            [
                {"label": "mean forward return", "values": means},
                {"label": "0.95 CI half-width", "values": errors, "error_of": "mean forward return"},
            ],
            params.period,
            len(frame),
            kind="bar_error",
            extra_caption=f"unconditional mean {numeric(unconditional)}; baseline is the period mean, not zero",
        )
        return {
            "n_obs": int(len(frame)),
            "n_units": int(len(frame)),
            "n_dropped": int(selection.n_dropped),
            "stats": stats,
            "table": bucket_rows,
            "chart": chart,
            "caveats": [
                "forward returns overlap across sessions at h>1, so the CI uses session blocks",
                "the baseline is the period's own unconditional mean; a bucket mean is not a strategy return",
                "no transaction costs, no slippage, no position sizing: this is not a backtest",
            ] + selection.caveats,
        }

    @registry.add("corr_matrix", CorrMatrixParams,
                  "Correlation matrix over 2-8 panel columns, pooled or averaged per ticker.",
                  chart_kind="heatmap")
    def corr_matrix(context: ToolContext, params: CorrMatrixParams) -> dict[str, Any]:
        columns = [metric_column(context, name) for name in params.metrics]
        selection = context.select(period=params.period)
        frame = selection.frame
        if len(frame) < 8:
            return {"n_obs": 0, "n_units": int(len(frame)),
                    "stats": {"note": "too few units for a matrix"}, "caveats": selection.caveats}

        matrix: dict[str, dict[str, float | None]] = {}
        if params.level == "per_ticker":
            for left in columns:
                matrix[left] = {}
                for right in columns:
                    values = []
                    for _, group in frame.groupby("ticker", observed=True):
                        rho, _ = spearman(values_of(group, left).to_numpy(), values_of(group, right).to_numpy())
                        if np.isfinite(rho):
                            values.append(rho)
                    matrix[left][right] = numeric(float(np.mean(values))) if values else None
        else:
            standardised = {column: within_ticker_z(frame, column) for column in columns}
            for left in columns:
                matrix[left] = {}
                for right in columns:
                    rho, _ = spearman(standardised[left].to_numpy(), standardised[right].to_numpy())
                    matrix[left][right] = numeric(rho)

        table = [{"metric": left, **{right: matrix[left][right] for right in columns}} for left in columns]
        chart = ChartSpec(
            kind="heatmap",
            title=f"Correlation matrix ({params.level})",
            x={"label": "metric", "values": columns},
            series=[{"label": left, "values": [matrix[left][right] for right in columns]} for left in columns],
            caption=f"Spearman, {params.level}; period={params.period}; n_units={len(frame)}",
        )
        return {
            "n_obs": int(len(frame)),
            "n_units": int(len(frame)),
            "stats": {"level": params.level, "matrix": matrix, "period": params.period},
            "table": table,
            "chart": chart,
            "caveats": [
                "a matrix of pairwise correlations is descriptive; no multiplicity correction is applied to it",
                "within-ticker standardisation is applied for the pooled level",
            ] + selection.caveats,
        }

    @registry.add("spike_days", SpikeParams,
                  "Ticker-sessions whose metric is a robust-z outlier against their own ticker.",
                  chart_kind="line")
    def spike_days(context: ToolContext, params: SpikeParams) -> dict[str, Any]:
        column = metric_column(context, params.metric)
        selection = context.select(period=params.period, scope=params.scope, ticker=params.ticker,
                                   require=(column,))
        frame = selection.frame.sort_values(["ticker", "session_idx"]).copy()
        if len(frame) == 0:
            return {"n_obs": 0, "n_units": 0, "stats": {"metric": column, "note": "no usable units"},
                    "caveats": selection.caveats}

        frame["cell_robust_z"] = frame.groupby("ticker", observed=True)[column].transform(
            lambda series: robust_z(series.to_numpy())
        )
        spikes = frame[frame["cell_robust_z"].abs() >= params.z_threshold]
        rows = [
            {
                "ticker": row.ticker,
                "date": str(row.session_date)[:10],
                "value": numeric(getattr(row, column)),
                "robust_z": numeric(row.cell_robust_z, 3),
                "tier": context.tiers.get(row.ticker),
            }
            for row in spikes.sort_values("cell_robust_z", key=lambda s: s.abs(), ascending=False).itertuples()
        ]
        stats = {
            "metric": column,
            "z_threshold": params.z_threshold,
            "n_spikes": int(len(spikes)),
            "spike_share": numeric(len(spikes) / len(frame), 4),
            "max_abs_z": numeric(frame["cell_robust_z"].abs().max()),
            "spikes_by_ticker": {str(k): int(v) for k, v in spikes["ticker"].value_counts().items()},
        }
        if params.scope == "ticker" and params.ticker:
            series_frame = frame
            labels = [str(value)[:10] for value in series_frame["session_date"]]
            values = [numeric(v) for v in values_of(series_frame, column)]
            annotations = [
                {"x": str(row.session_date)[:10], "y": numeric(getattr(row, column)), "label": f"z={numeric(row.cell_robust_z, 2)}"}
                for row in spikes.itertuples()
            ]
            chart = line_chart(f"{column} for {params.ticker}, robust-z spikes marked", "session", labels,
                              [{"label": column, "values": values}], params.period, len(frame),
                              annotations=annotations)
        else:
            per_session = frame.groupby("session_idx", observed=True)["cell_robust_z"].apply(
                lambda series: int((series.abs() >= params.z_threshold).sum())
            ).sort_index()
            labels = [
                str(value)[:10]
                for value in frame.drop_duplicates("session_idx").sort_values("session_idx")["session_date"]
            ]
            chart = line_chart(f"Spike count per session ({column})", "session", labels,
                              [{"label": "spikes", "values": [int(v) for v in per_session]}],
                              params.period, len(frame))
        return {
            "n_obs": int(len(frame)),
            "n_units": int(len(frame)),
            "stats": stats,
            "table": rows[:50],
            "chart": chart,
            "caveats": [
                "robust z (median/MAD) within the period, so a ticker-session is compared with its own ticker",
                "a spike is an outlier, not a significant event; several spikes are calendar-wide news days",
                "table is capped at 50 rows",
            ] + selection.caveats,
        }

    @registry.add("attention_vs_price", TickerPairParams,
                  "Specific (A3) and market-wide (A4) attention beside the close for one ticker.",
                  chart_kind="dual_line")
    def attention_vs_price(context: ToolContext, params: TickerPairParams) -> dict[str, Any]:
        selection = context.select(period=params.period, scope="ticker", ticker=params.ticker,
                                   require=("close",))
        frame = selection.frame.sort_values("session_idx")
        if len(frame) == 0:
            return {"n_obs": 0, "n_units": 0, "stats": {"ticker": params.ticker},
                    "caveats": selection.caveats + [f"no usable sessions for {params.ticker}"]}
        labels = [str(value)[:10] for value in frame["session_date"]]
        rho_a3, _ = spearman(values_of(frame, "n_texts_own").to_numpy(), values_of(frame, "close").to_numpy())
        rho_a4, _ = spearman(values_of(frame, "n_texts_shared").to_numpy(), values_of(frame, "close").to_numpy())
        chart = line_chart(
            f"{params.ticker}: specific vs market-wide attention and close",
            "session",
            labels,
            [
                {"label": "n_texts_own (A3)", "values": [numeric(v) for v in frame["n_texts_own"]]},
                {"label": "n_texts_shared (A4)", "values": [numeric(v) for v in frame["n_texts_shared"]]},
                {"label": "close", "values": [numeric(v) for v in frame["close"]], "axis": "right"},
            ],
            params.period,
            len(frame),
            extra_caption="A1 (raw rows) is not plotted alone; A3 is the ticker-specific measure",
            kind="dual_line",
        )
        return {
            "n_obs": int(frame["n_tweets"].sum()),
            "n_units": int(len(frame)),
            "stats": {
                "ticker": params.ticker,
                "tier": context.tiers.get(params.ticker),
                "sessions": int(len(frame)),
                "mean_n_texts_own": numeric(frame["n_texts_own"].mean()),
                "mean_n_texts_shared": numeric(frame["n_texts_shared"].mean()),
                "spearman_a3_vs_close": numeric(rho_a3),
                "spearman_a4_vs_close": numeric(rho_a4),
            },
            "chart": chart,
            "caveats": [
                "descriptive dual axis: the two axes have different units, so a visual overlap is not an effect",
                "news drives both attention and price, so neither direction is identified here",
            ] + selection.caveats,
        }

    @registry.add("sentiment_vs_price", TickerPairParams,
                  "Mean daily sentiment beside the close for one ticker.",
                  chart_kind="dual_line")
    def sentiment_vs_price(context: ToolContext, params: TickerPairParams) -> dict[str, Any]:
        selection = context.select(period=params.period, scope="ticker", ticker=params.ticker,
                                   require=("close", "sent_net"))
        frame = selection.frame.sort_values("session_idx")
        if len(frame) == 0:
            return {"n_obs": 0, "n_units": 0, "stats": {"ticker": params.ticker},
                    "caveats": selection.caveats + [f"no scored sessions for {params.ticker}"]}
        labels = [str(value)[:10] for value in frame["session_date"]]
        rho, p_value = spearman(values_of(frame, "sent_net").to_numpy(), values_of(frame, "close").to_numpy())
        rho_next, _ = spearman(
            values_of(frame, "sent_net").to_numpy()[:-1],
            values_of(frame, "fwd_ret_1").to_numpy()[:-1],
        )
        chart = line_chart(
            f"{params.ticker}: sentiment and close",
            "session",
            labels,
            [
                {"label": "sent_net", "values": [numeric(v) for v in frame["sent_net"]]},
                {"label": "close", "values": [numeric(v) for v in frame["close"]], "axis": "right"},
            ],
            params.period,
            len(frame),
            extra_caption="co-movement only; the forward-return relation is tested by bucket_forward_returns",
            kind="dual_line",
        )
        return {
            "n_obs": int(frame["n_tweets"].sum()),
            "n_units": int(len(frame)),
            "stats": {
                "ticker": params.ticker,
                "sessions": int(len(frame)),
                "mean_sent_net": numeric(frame["sent_net"].mean()),
                "spearman_sent_net_vs_close": numeric(rho),
                "p_value_contemporaneous": numeric(p_value),
                "spearman_sent_net_vs_fwd_ret_1": numeric(rho_next),
            },
            "chart": chart,
            "caveats": [
                "the contemporaneous correlation is descriptive co-movement and is never called predictive",
                f"{int(frame['n_tweets'].sum())} tweets are aggregated into {len(frame)} ticker-sessions, which is the real sample size",
            ] + selection.caveats,
        }


def _direction(x_col: str, y_col: str, lag: int) -> str:
    """Describe the time direction of a correlation in words, so a caption cannot mislead."""
    if y_col.startswith("fwd_ret_"):
        horizon = y_col.rsplit("_", 1)[-1]
        return (
            f"{x_col} on day d{'lagged' if lag else ''} against the forward {horizon}-session return "
            "from day d: a predictive-style arrangement, still not causal evidence"
        )
    if lag:
        return f"{x_col} shifted {lag} session(s) against {y_col} on the same day: a lead-lag co-movement"
    return f"{x_col} and {y_col} measured on the same session: co-movement, no direction implied"


def _bootstrap_p_value(
    values: np.ndarray,
    signs: np.ndarray,
    session_ids: np.ndarray,
    block_sessions: int,
    resamples: int,
    seed: int,
) -> float:
    """Two-sided bootstrap p-value for a weighted mean being zero.

    The share of resamples on each side of zero, doubled and capped at 1. It is a bootstrap
    p-value, not a t-test p-value: it makes no normal-approximation assumption, which is why it is
    the one reported beside a block-bootstrap CI.
    """
    from ..stats.inference import _block_index_groups  # local: the block layout is an implementation detail

    mask = np.isfinite(values) & np.isfinite(signs) & (signs != 0)
    values, signs = values[mask], signs[mask]
    if len(values) < 4:
        return float("nan")
    blocks = _block_index_groups(np.asarray(session_ids)[mask], max(1, block_sessions))
    if len(blocks) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    for i in range(resamples):
        index = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), size=len(blocks))])
        means[i] = (values[index] * signs[index]).mean()
    below = float(np.mean(means <= 0))
    above = float(np.mean(means >= 0))
    return round(min(1.0, 2 * min(below, above)), 6)
