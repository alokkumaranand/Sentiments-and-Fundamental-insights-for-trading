# Step 0 — Dataset Profile (verified on the real files)

**Status:** complete · replaces every "assumed / to be verified" item in spec §2
**Method:** read-only profiling with `pandas` 2.x over the two files present in the workspace root.
Nothing was written to, moved, or modified in the raw inputs.

> This file is the evidence for §2 of `SPEC.md`. Every claim here was produced by running code
> against the actual files, not copied from the Kaggle description. Where the old spec's assumption
> was wrong, the correction is marked **CORRECTION**.

---

## 1. Inputs and hashes

| File | Bytes | SHA-256 |
|---|---|---|
| `stock_tweets.csv.zip` | 6,754,415 | `C5331333F4643AA06B3C637EF770CCD940F770C69ECE0006931F6DDF91151810` |
| `stock_yfinance_data.csv` | 712,372 | `59BD6477B7CCD942539BA4B598C8C58C790955207B8ECFBD61158A27C8DEBEAD` |

Hashes are of the containers as delivered; the loader also hashes the decompressed tweet CSV
member, so a repackaged zip with identical contents is still detected as the same data.

**CORRECTION — the tweets file is a ZIP, not a CSV.**
The old spec assumed `stock_tweets.csv`. The actual artifact is `stock_tweets.csv.zip`, containing a
single member `stock_tweets.csv` (18,037,306 bytes uncompressed, ratio ≈ 2.67×).
Consequence: the loader must read the CSV **directly out of the zip** (`zipfile` + `BytesIO`).
It must not extract into `data/raw/`, so the raw directory stays exactly as delivered.

---

## 2. Tweet file — actual schema

```
Date,Tweet,Stock Name,Company Name
```

| Column | dtype as read | Nulls | Notes |
|---|---|---|---|
| `Date` | object (string) | 0 | RFC-3339 with explicit UTC offset, e.g. `2022-09-29 23:41:16+00:00` |
| `Tweet` | object | 0 | clean UTF-8; median 152 chars, max 462 |
| `Stock Name` | object | 0 | ticker symbol only, 25 distinct |
| `Company Name` | object | 0 | display name; **exactly one per ticker** (no case/whitespace variants) |

- **Rows:** 80,793 · **Columns:** 4 · **Exact duplicate rows:** 0
- **Column names verified** exactly as assumed (casing and spacing correct). No trailing-space headers.
- File is stored newest-first (first row `2022-09-29 23:41:16+00:00`).

### 2.1 Dates and timezone — verified

- **Every** one of 80,793 `Date` values ends in `+00:00`. No naive timestamps, no missing offsets.
- Parse with `pd.to_datetime(..., utc=True)`: **0** unparseable values.
- Range: `2021-09-30 00:06:02+00:00` → `2022-09-29 23:41:16+00:00`.
- This matches the assumed 30 Sep 2021 → 30 Sep 2022 window.

### 2.2 Session alignment — measured impact

Trading calendar taken from the price file (252 sessions, see §3). Tweets bucketed by
**US/Eastern** date and 16:00 ET close:

| Bucket | Rows | Share |
|---|---|---|
| Posted before 16:00 ET on a trading date | 49,501 | 61.3% |
| Posted at/after 16:00 ET | 31,292 | 38.7% |
| Posted on a non-trading ET date (weekend/holiday) | 17,866 | **22.1%** |

- 15,984 rows fall on a Saturday or Sunday (19.8%).
- Combined, **≈49% of all tweets roll forward** to the next trading session under the alignment rule.
  This is not an edge case; it is the normal path for half the corpus and is why the alignment rule
  must be table-driven and tested first.

### 2.3 Coverage per ticker — severely uneven

| Ticker | Tweets | Share | Ticker | Tweets | Share |
|---|---:|---:|---|---:|---:|
| TSLA | 37,422 | 46.3% | DIS | 635 | 0.8% |
| TSM | 11,034 | 13.7% | BA | 399 | 0.5% |
| AAPL | 5,056 | 6.3% | COST | 393 | 0.5% |
| AMZN | 4,089 | 5.1% | INTC | 315 | 0.4% |
| MSFT | 4,089 | 5.1% | KO | 310 | 0.4% |
| PG | 4,089 | 5.1% | CRM | 233 | 0.3% |
| NIO | 3,021 | 3.7% | XPEV | 225 | 0.3% |
| META | 2,751 | 3.4% | ENPH | 216 | 0.3% |
| AMD | 2,227 | 2.8% | ZS | 193 | 0.2% |
| NFLX | 1,727 | 2.1% | VZ | 123 | 0.2% |
| GOOG | 1,291 | 1.6% | BX | 50 | 0.1% |
| PYPL | 843 | 1.0% | F | 31 | 0.04% |
| | | | NOC | 31 | 0.04% |

**Consequences that change the design:**
- **CORRECTION — pooling is dominated by one ticker.** 46.3% of the corpus is TSLA. A "pooled"
  statistic is close to a TSLA statistic. Pooling must be within-ticker standardized (already
  planned) **and** TSLA-excluded / TSLA-only variants must be shown side by side. A single pooled
  number will not be reported on its own anywhere in the UI.
- The bottom four tickers (BX 50, F 31, NOC 31, VZ 123) cannot support per-ticker statistics.
  Coverage tiering is required (§2.6).

### 2.4 Duplicate text and cross-ticker fan-out — the most important structural finding

- **Distinct tweet texts: 64,479** from 80,793 rows.
- **6,948 texts appear under more than one ticker**, covering **23,060 rows (28.5%)**.
- Only **257 rows** are an exact `(ticker, text)` repeat. So the redundancy is almost entirely
  *fan-out*: one text mentioning several tickers is emitted once per ticker.

Share of each ticker's tweets whose text also appears under another ticker:

| Ticker | Share of rows that are shared text | Ticker | Share |
|---|---:|---|---:|
| AMZN | **100.0%** | AMD | 43.0% |
| MSFT | **100.0%** | DIS | 40.3% |
| PG | **100.0%** | META | 40.0% |
| NOC | **100.0%** | NFLX | 39.7% |
| F | **100.0%** | ENPH | 39.4% |
| VZ | 65.0% | ZS | 34.7% |
| BA | 63.4% | NIO | 15.4% |
| KO | 62.9% | TSLA | 6.6% |
| INTC | 61.3% | TSM | 1.2% |
| XPEV | 56.4% | | |

**Interpretation (this is why AMZN, MSFT and PG all have exactly 4,089 tweets, and F and NOC both
have exactly 31):** those five tickers' corpora are *entirely* generic multi-cashtag text. A
per-ticker sentiment or attention number for them measures the generic market-chatter pool,
**not** that company. The identical counts are the giveaway: a ticker whose row count exactly equals
another's is receiving the same shared text, not its own audience.
- TSLA (6.6%) and TSM (1.2%) are genuinely ticker-specific corpora; NIO (15.4%) is the third.
- This single fact must be a first-class feature (`own_text_share`), a documented limitation, and a
  visible badge — not a footnote.

### 2.5 Text quality

| Property | Rows | Share |
|---|---:|---:|
| Contains a `$cashtag` | 60,725 | 75.2% |
| Contains a URL (`t.co`) | 37,779 | 46.8% |
| Contains a `#hashtag` | 12,342 | 15.3% |
| Starts with `@` (reply-shaped) | 8,040 | 10.0% |
| Contains an HTML entity (`&amp;`, `&gt;`, `&#…`) | 7,745 | 9.6% |
| Contains ≥4 cashtags **or** a URL | 45,069 | **55.8%** |

- **Encoding is clean.** The file decodes as strict UTF-8 with no errors. (An earlier terminal
  display of `heardâ€œthey` was a Windows console rendering artifact, **not** double-encoded data.)
  Emoji are real emoji.
- HTML entities are real and must be unescaped in cleaning (`html.unescape`).
- **Promo/spam is the majority pattern.** Top repeated texts are promotional blasts listing a dozen
  tickers, e.g. 14 copies of `100% FREE 👇 https://t.co/… $SPX $QQQ $SPY $AAPL $BABA $NFLX $BA $COST $TSLA …`.
  A promo/pump filter is required, and `promo_share` is a reported metric, not a silent drop.

### 2.6 Coverage tiering (derived)

Row grain used below is the **ticker × ET date** pair, before session rolling: 5,906 pairs.

| Statistic | Value |
|---|---:|
| Median tweets per ticker-day | 5 |
| Mean tweets per ticker-day | 13.7 |
| Ticker-days with < 2 tweets | 1,414 (23.9%) |
| Ticker-days with < 3 tweets | 2,137 (36.2%) |
| Ticker-days with < 5 tweets | 2,950 (**50.0%**) |
| Ticker-days per ticker | min 30 · median 242 · max 366 ᵃ |

ᵃ The max exceeds 252 because this grain counts distinct *calendar* ET dates that carry tweets —
including weekends and holidays — before the session-rolling step collapses them onto trading days.
After rolling, no ticker can exceed 252 sessions.

- **Half of all ticker-days carry fewer than 5 tweets.** A minimum-coverage threshold in the 3–5
  range will exclude roughly a third to a half of ticker-days. This is the single largest driver of
  the honest sample size and must be configurable and always reported as a count, never applied
  silently.
- The thinnest tickers (F, NOC: 31 ticker-days; BX: 50) have tweets on only a small fraction of the
  252 sessions. This is thin *text* coverage, not missing price rows — the price file is complete for
  every ticker (§3).

---

## 3. Price file — actual schema

```
Date,Open,High,Low,Close,Adj Close,Volume,Stock Name
```

| Property | Verified value |
|---|---|
| Shape | 6,300 rows × 8 columns |
| Columns | exactly as the old spec assumed, `Adj Close` spelled with a space |
| Nulls | **0** in every column |
| Duplicate `(ticker, date)` | 0 |
| Weekend rows | 0 (trading days only) |
| Date range | 2021-09-30 → 2022-09-29 |
| Trading days | **252**, identical for all 25 tickers (std = 0) |
| Distinct tickers | 25 |
| `Adj Close == Close` | 3,539 / 6,300 rows |
| max abs(`Adj Close` − `Close`) | 7.5978 |

- **Adj Close matters.** It differs from Close on 2,761 rows (dividend payers: VZ, BX, INTC, F, KO, …)
  by up to $7.60. The spec's "prefer `adj_close`" rule is correct and load-bearing; the chosen series
  is logged per ticker.

### 3.1 Split continuity — tested, because returns depend on it

The dataset window contains real splits (TSLA 3:1 in Aug 2022, AMZN 20:1 in Jun 2022). An unadjusted
`Close` would show a −95% one-day "return" on the split date and would corrupt every return,
volatility and bucket statistic downstream. Tested directly:

| Ticker | `Close` first → last | Split present in window | Largest 1-day `Close` move |
|---|---|---:|---:|
| AMZN | 164.25 → 114.80 | 20:1 (Jun 2022) | 14.1% (2022-04-29) |
| TSLA | 258.49 → 268.21 | 3:1 (Aug 2022) | 13.5% (2022-01-03) |
| GOOG | 133.27 → 98.09 | 20:1 (Jul 2022) | 7.7% |
| AAPL | 141.50 → 142.48 | — | 7.0% |
| MSFT | 281.92 → 237.50 | — | 6.7% |

**Conclusion: `Close` is already split-adjusted.** The prices are on a post-split basis (AMZN at
$164.25 on 2021-09-30, not $3,285), and across all 6,300 rows the six largest one-day moves are
ordinary news moves — NFLX −35.2% (2022-04-20 subscriber loss), XPEV +29.6% and NIO +25.6%
(2022-03-16 China stimulus rally), META −26.4% (2022-02-03), ENPH +24.7%, PYPL −24.6%. No 20:1 or 3:1
discontinuity appears anywhere.

**Therefore no split handling is needed in v0**, and `Adj Close` differs from `Close` purely by
dividend adjustment — confirmed by the relative gap: max 6.3% (VZ), 5.2% (BX), 4.5% (INTC), which is
the size of one year of dividends for those names, not a split.

Reproduce with `eda profile --check-splits`, which asserts `max |1-day close return| < 0.5` for every
ticker (the largest observed is 0.352) and fails loudly if a future data refresh breaks the
assumption.
- Prices are fully rectangular: no per-ticker gaps, no holidays-as-rows. **The price file is
  therefore the authoritative trading calendar** — no exchange-calendar dependency is required.
  This keeps the dependency surface small and the alignment fully deterministic.

---

## 4. Cross-file reconciliation

| Check | Result |
|---|---|
| Tickers only in prices | **none** |
| Tickers only in tweets | **none** |
| Ticker sets identical | ✅ 25 = 25 |
| `Stock Name` format matches across files | ✅ bare symbols, e.g. `TSLA` |
| Company Name unique per ticker | ✅ (0 tickers with >1 name) |
| Date-window overlap | ✅ full — both end 2022-09-29 |
| Blocking errors under spec §4.1 | **none** |

The assumed join key works as-is. No ticker mapping table is needed for v0 (an asset-mapping
audit remains an upgrade-path item).

---

## 5. Discovery / holdout split — resolved

Chronological 70/30 on the 252-session calendar:

- **Discovery:** 2021-09-30 → **2022-06-10** — 176 sessions
- **Holdout:** next session → 2022-09-29 — 76 sessions

This resolves open decision §13.4 of the old spec. The split is a session *count*, not a date
literal, so it stays correct if the data window changes; the resolved dates are logged in the run
manifest.

---

## 6. What this changes in the spec

| # | Finding | Change made |
|---|---|---|
| 1 | Tweets arrive as `.zip` | Loader reads the CSV member in-memory; no extraction |
| 2 | 46.3% of tweets are TSLA | Pooled statistics never reported alone; TSLA split out |
| 3 | 28.5% of rows are cross-ticker fan-out; 5 tickers are 100% generic | `own_text_share` feature, 3-tier coverage scheme, visible badge, documented limit |
| 4 | 55.8% of tweets are promo/URL-shaped | Deterministic promo filter + `promo_share` metric, reported not dropped |
| 5 | 9.6% contain HTML entities | `html.unescape` in cleaning, with a row-count log |
| 6 | ~49% of tweets roll forward a session; 22.1% are non-trading-day | Alignment rule is table-driven and tested first |
| 7 | Half of ticker-days have <5 tweets | Configurable min-coverage; exclusions always counted and reported |
| 8 | Price file is a clean rectangle | Price dates are the trading calendar; no calendar dependency |
| 9 | Adj Close ≠ Close on 44% of rows | `adj_close` is the default return basis, logged |
| 10 | Split resolvable now | Discovery/holdout fixed at 2022-06-10 |

**No blocking errors found.** Every item in the old §2 verification checklist is now answered.
