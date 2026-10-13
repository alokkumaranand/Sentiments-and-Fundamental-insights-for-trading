# Agentic EDA Demo — System Design Spec

Simplified first version of the Agentic Market Research & Insight Discovery platform
Stock tweets + prices → agent-induced features → deterministic EDA → insights presented in the terminal

**Draft 0.2 · revised after review of draft 0.1 · dataset assumptions now verified**

Companion document: `STEP0_PROFILE.md` — the real, measured dataset profile. Draft 0.1 §2 was written
without the files and said so; every item in it is now answered there. This spec depends on it.

---

## 0. What changed from draft 0.1

| # | Source of change | Change |
|---|---|---|
| 1 | Your review | **The LLM moves into the initial layer.** It is no longer only an explorer on top. A Feature Induction Agent reads a capped sample, proposes the sentiment vocabulary / relevance rules / negation patterns, and a deterministic validator accepts or rejects each proposal against a disjoint slice. Deterministic code then scores the whole corpus. See §4.3–4.5. |
| 2 | Your review | **Only two top-layer insight families: Sentiment and Attention.** Both are deductions on top of lower-layer patterns. Each is a small, named, conservative measure set (7 each), fully specified with methodology in §4.5 and §4.6. |
| 3 | Your review | **New information is first-class.** A documented incremental path: arriving tweets are scored deterministically with the frozen lexicon; a coverage-drift check decides when re-induction is warranted. See §7.3. |
| 4 | Your review | **No second UI.** Streamlit and Plotly are dropped. Everything non-visual prints as a terminal overview and as Markdown; charts are chosen from a numbered terminal menu. See §4.11 and §5. |
| 5 | Your review | **`.venv/` and `.env` are in the repo layout**, with `.env.example`, `.gitignore`, and pinned requirements. LLM base URL / name / key live in `.env`; non-secret wiring lives in `config/settings.yaml`. See §6. |
| 6 | Your review | **Five build phases, each ending at a human confirmation gate.** Phase 1 is where you enter the LLM details and run the live smoke test. See §11. |
| 7 | Your review | **Code-explanation convention made explicit** — short, straightforward docstrings and "why" comments required. See §15. |
| 8 | Verified data | Ten concrete corrections from `STEP0_PROFILE.md`: zip container, TSLA = 46.3% of corpus, 28.5% cross-ticker fan-out, 55.8% promo-shaped text, 9.6% HTML entities, ~49% of tweets roll a session, half of ticker-days below 5 tweets, price file as trading calendar, Adj Close load-bearing, split fixed at 2022-06-10. |
| 9 | Retained | Three design rules, `ToolResult`/`Finding` contracts, deterministic claim checker, past-only features, chronological holdout, block bootstrap, BH adjustment, rule-assigned status, run manifest. These were solid and are unchanged in substance. |

---

## 1. Purpose and scope

### 1.1 Goal

Build the smallest end-to-end demo that shows one clean loop:

```
dataset → LLM-induced, deterministically-applied features → deterministic EDA
        → agents choose follow-ups and interpret → findings presented in the terminal
```

The demo does standard exploratory data analysis (univariate, temporal, bivariate, conditional,
anomaly) on tweet text and daily prices, with a thin agent layer both **below** (feature induction)
and **above** (exploration and interpretation) the deterministic core.

It is a showcase of the pattern, not a trading system.

### 1.2 In scope for v0

- Load and profile the real dataset read-only (zip-aware loader).
- Session alignment, canonical tables, and a ticker-session panel.
- **Feature Induction Agent** (LLM) that proposes lexicons and rules on a capped sample.
- **Deterministic validator** that accepts or rejects each proposal on a disjoint slice.
- **Two feature families: sentiment (7 measures) and attention (7 measures).**
- A library of typed, deterministic EDA tools.
- A fixed baseline playbook that runs with no LLM.
- An Explorer agent that picks follow-up analyses; a Presenter agent that writes the narrative.
- A deterministic claim checker so agents cannot invent numbers.
- Terminal-first presentation: Markdown report + numbered chart menu.
- An incremental update path for new data.

### 1.3 Out of scope for v0

Insight registry with versioning · full validation-status lifecycle · multi-agent debate ·
economic/transaction-cost validation · regime discovery · live data feeds · **any buy/sell language** ·
Streamlit or any web UI · FinBERT or any transformer sentiment model · author-level features
(the dataset has no author column — see §2.4).

### 1.4 Design rules

1. **Code computes; agents choose and explain.** Every number shown to a user comes from a tool
   result with an ID.
2. **Agents act only through registered tools.** No shell, no arbitrary code, no raw dataset access
   beyond a capped tweet-sampling tool.
3. **The demo must run without an LLM key.** Agents are an enhancement layer; a no-key run still
   produces a complete, templated, honest report.
4. **The LLM induces; deterministic code applies.** *(new)* No LLM ever scores a row, assigns a
   status, or writes a number. Its only low-level output is a reviewable, hashed vocabulary/ruleset
   that plain code then applies at scale.
5. **New data reuses frozen artifacts before it triggers new reasoning.** *(new)* Incoming tweets
   are scored with the existing lexicon; re-induction happens only when a drift check says the
   frozen vocabulary no longer covers the text.

---

## 2. Dataset: verified facts

Full evidence, per-ticker tables and the profiling method are in `STEP0_PROFILE.md`.
This section states only what the build depends on.

### 2.1 Files

| File | SHA-256 (as delivered) | Role |
|---|---|---|
| `stock_tweets.csv.zip` → member `stock_tweets.csv` | `C5331333…1810` | Text events, one row per tweet |
| `stock_yfinance_data.csv` | `59BD6477…BEAD` | Daily prices per ticker |

The loader reads the CSV **directly from the zip** in memory and never extracts into `data/raw/`.

### 2.2 Schemas — confirmed, no discrepancies

- tweets: `Date, Tweet, Stock Name, Company Name` — 80,793 rows, **0 nulls, 0 duplicate rows**
- prices: `Date, Open, High, Low, Close, Adj Close, Volume, Stock Name` — 6,300 rows, **0 nulls**
- `Date` in tweets is RFC-3339 with an explicit `+00:00` offset on **all 80,793 rows**; 0 unparseable.
- Tickers match exactly across both files (25 = 25). `Company Name` is unique per ticker.
- Window: **2021-09-30 → 2022-09-29**.

### 2.3 Calendars

- Prices are a **complete rectangle**: 25 tickers × exactly 252 sessions, no gaps, no weekend rows,
  no duplicate `(ticker, date)` pairs, null-free.
- **Therefore the price dates are the authoritative trading calendar.** No `exchange_calendars`
  or `pandas_market_calendars` dependency; alignment stays deterministic and dependency-light.
- Adj Close differs from Close on 2,761 rows (max $7.60, dividend payers). `adj_close` is the
  default return basis; the choice is logged per ticker in the manifest.
- **Split continuity verified.** The window contains real splits (AMZN 20:1, TSLA 3:1, GOOG 20:1) and
  both price columns are already on a post-split basis — the largest one-day `Close` move in all
  6,300 rows is NFLX −35.2% (2022-04-20, its real subscriber-loss crash), with no 20:1 or 3:1
  discontinuity anywhere. So `Adj Close` differs from `Close` purely by dividends, and **no split
  handling is needed in v0**. `eda profile --check-splits` asserts this and fails loudly if a future
  data refresh breaks it (evidence in `STEP0_PROFILE.md` §3.1).

### 2.4 Structural limits (stated in the terminal overview and in every report)

| Limit | Measured |
|---|---|
| One regime only | 252 sessions, includes the 2022 drawdown; findings do not generalize |
| Per-ticker noise | ~252 sessions per ticker, and half of all ticker-days carry <5 tweets |
| Not 80K independent observations | the independent unit is the **ticker-day** (5,906 pairs pre-roll), not the tweet |
| **Pooling is dominated by TSLA** | TSLA = 46.3% of all tweets. A pooled number is nearly a TSLA number |
| **28.5% of rows are cross-ticker fan-out** | 6,948 texts appear under >1 ticker, covering 23,060 rows |
| **5 tickers are entirely generic text** | AMZN, MSFT, PG, F, NOC: **100%** of their tweets also appear under another ticker. Their "sentiment" is the generic chatter pool, not the company |
| **Promo/spam is the majority pattern** | 55.8% of rows contain a URL or ≥4 cashtags; top repeated texts are promo blasts |
| **~49% of tweets roll a session** | 38.7% posted at/after 16:00 ET; 22.1% on a non-trading ET date |
| No author field | user-level, bot-detection and unique-author metrics are **impossible**; not deferred, unavailable |
| Timestamps ≠ information arrival | the tweet time is not the time the market learned it |

### 2.5 Coverage tiers — a first-class concept

Derived from the **measured** per-ticker own-text share (fraction of a ticker's tweets whose text
appears under no other ticker). Tier is written into every result, so a number can never be read out
of context.

| Tier | Rule (own-text share) | Tickers (share) | How they may be used |
|---|---|---|---|
| **A — specific** | ≥ 0.80 | TSM 0.99 · TSLA 0.93 · NIO 0.85 — **3 of 25** | per-ticker and pooled analyses |
| **B — mixed** | 0.10 – 0.80 | ZS 0.65 · ENPH 0.61 · NFLX 0.60 · META 0.60 · DIS 0.60 · AMD 0.57 · AAPL 0.54 · PYPL 0.54 · COST 0.53 · GOOG 0.52 · CRM 0.49 · BX 0.48 · XPEV 0.44 · INTC 0.39 · KO 0.37 · BA 0.37 · VZ 0.35 — **17 of 25** | descriptive; per-ticker inference only with `own_text_only` on and the fan-out share printed beside it |
| **C — generic** | < 0.10 | AMZN 0.00 · MSFT 0.00 · PG 0.00 · F 0.00 · NOC 0.00 — **5 of 25** | **descriptive coverage only**; excluded from sentiment and attention inference; shown with an explicit badge |

Own-text share is computed from the data, never guessed, and the thresholds are the only tunable
part. Tier assignment is printed in the overview so the rule is never hidden.

**Read the consequence plainly:** only 3 of 25 tickers are ticker-specific enough for unrestricted
per-ticker inference, and 5 are not ticker-specific at all. That is a limit of this dataset, not of
the pipeline — and it is why the "pooled" and "TSLA-excluded pooled" variants in §4.8 are mandatory
rather than optional.

Usable-session counts (sessions meeting `min_tweets_per_day`) are computed per ticker in Phase 2 and
reported alongside the tier; they are deliberately not asserted here, because they depend on a
threshold that is still open (§13.5).

### 2.6 Blocking errors and warnings

**Blocking** (stop with a clear message): unparseable dates · missing required columns · no ticker
overlap between the files · **tweet corpus containing no text unique to any single ticker** ·
**fewer than 2 coverage-tier-A tickers**.

**Warnings** (carried forward into the report and the manifest): everything in §2.4, plus the
per-ticker coverage table, the excluded-ticker-days count, the promo-share table, and any ticker
whose discovery or holdout slice has fewer than a configured minimum of usable ticker-days.

**No blocking error was found in the current data.**

### 2.7 Discovery / holdout split — resolved

Chronological 70/30 on the session calendar: **discovery 2021-09-30 → 2022-06-10 (176 sessions)**,
**holdout 2022-06-13 → 2022-09-29 (76 sessions)**. Configurable as a *fraction*, not a date literal.

---

## 3. Architecture

Modular monolith, one repo, one dataframe library (pandas; the dataset is small).
Stages communicate through files in `artifacts/`, so each stage runs alone or in the pipeline.

```
 data/raw/                       ┌──────────────────────────────────────────────┐
   stock_tweets.csv.zip ────────►│ A · INGEST & PROFILE          (no LLM)       │
   stock_yfinance_data.csv ─────►│   hash inputs · read-only · profile.json     │
                                 │   blocking errors / warnings                 │
                                 └───────────────────┬──────────────────────────┘
                                                     ▼
                                 ┌──────────────────────────────────────────────┐
                                 │ B · CANONICAL TABLES + ALIGNMENT (no LLM)    │
                                 │   tweets · prices · daily_panel              │
                                 │   session roll · promo flag · own_text flag  │
                                 └───────────────────┬──────────────────────────┘
                                                     ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │ C · FEATURE INDUCTION            (LLM · runs once, cached, optional)   │
        │   capped stratified sample → LexiconProposal                           │
        │   → deterministic validator on a DISJOINT slice → accepted subset      │
        │   → frozen, hashed lexicon  artifacts/lexicons/<hash>.json             │
        └───────────────────┬────────────────────────────────────────────────────┘
                            ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │ D · DETERMINISTIC FEATURE LAYER  (no LLM · applies the frozen lexicon)  │
        │   SENTIMENT family (7)   ·   ATTENTION family (7)                      │
        │   price features · past-only robust z-scores · quality flags           │
        └───────────────────┬────────────────────────────────────────────────────┘
                            ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │ E · EDA TOOL LIBRARY       (typed, pure, deterministic, Pydantic)      │
        │   ToolResult { result_id = hash(tool, params, data_version) }          │
        │   ChartSpec (declarative; renderer-agnostic)                           │
        └───────┬────────────────────────────────────────────┬───────────────────┘
                ▼                                            ▼
   ┌─────────────────────────┐                 ┌─────────────────────────────────┐
   │ F · BASELINE PLAYBOOK   │                 │ G · EXPLORER AGENT (LLM)        │
   │  fixed EDA, no LLM      │                 │  picks follow-up tools, ≤12     │
   └───────────┬─────────────┘                 └──────────────┬──────────────────┘
               └───────────────► ToolResult store ◄────────────┘
                              artifacts/runs/<run_id>/results/
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────────┐
                    │ H · FINDINGS + CLAIM CHECKER + STATUS RULES      │
                    │   deterministic; holdout re-run; BH adjustment   │
                    └───────────────────┬──────────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────────────┐
                    │ I · PRESENTER AGENT (LLM) → narrative + captions │
                    │   every numeric sentence carries [R:<result_id>] │
                    └───────────────────┬──────────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────────────┐
                    │ J · TERMINAL PRESENTATION                        │
                    │   rich overview · report.md · report.html        │
                    │   numbered chart menu → PNG · ASCII fallback     │
                    └──────────────────────────────────────────────────┘
```

The coordinator is plain Python, not an LLM. It sequences stages, enforces tool budgets, validates
agent output against schemas, and writes the run manifest.

---

## 4. Components

### 4.1 Ingest and profile

*(Architecture stage A — delivered in build Phase 0.)*

- Read raw inputs read-only. Never modify. Record SHA-256 of the zip container, the decompressed
  tweet CSV member, and the price CSV.
- Produce `profile.json` and a human-readable `docs/quality_report.md`: rows, columns, inferred
  types, min/max timestamps, distinct tickers, missingness, duplicate and cross-ticker-fanout
  counts, tweets per ticker and per ticker-day, price coverage per ticker, promo share, HTML-entity
  count, non-trading-day share, and warnings.
- Assert the §2.6 blocking conditions; warnings are carried forward into the report and manifest.

### 4.2 Canonical tables and alignment

Two canonical tables plus a derived panel.

| Table | Grain | Key columns |
|---|---|---|
| `tweets` | one row per tweet | `ticker, ts_utc, ts_original, text_clean, session_date, promo_flag, own_text_flag, shared_with_n, drop_reason` |
| `prices` | ticker × trading date | `ticker, date, open, high, low, close, adj_close, volume` |
| `daily_panel` | ticker × session date | attention, sentiment, price and quality columns below |

**Cleaning (each step logs before/after row counts):** `html.unescape` · whitespace collapse ·
URL replacement with a `__URL__` token (so promo detection survives) · cashtag/hashtag extraction ·
`promo_flag` = URL present **or** ≥4 cashtags · `own_text_flag` = text not shared with another
ticker · `shared_with_n` = number of tickers carrying that text.

Nothing is dropped for being low quality. Rows are flagged, counted, and reported. Drops happen
only inside a specific analysis, and that analysis states its `n_dropped`.

**Session alignment rule.** A tweet posted before the 16:00 US/Eastern close on a trading day belongs
to that session. A tweet posted at/after the close, or on a non-trading date, belongs to the next
trading session. Timestamps are stored as UTC with the original string preserved. Measured impact:
**~49% of tweets roll forward.** The rule is deliberately conservative — it may discard timing
information but cannot let a later tweet influence an earlier session.

**Return definitions** (on `adj_close`, else `close`, choice logged):

- *Contemporaneous* `ret_contemp(d)`: close(d−1) → close(d). Descriptive co-movement only, never
  labelled predictive.
- *Forward* `fwd_ret_h(d)`: close(d) → close(d+h), h ∈ {1, 2, 5}. The only return used in
  predictive-style analysis.

### 4.3 Feature Induction Agent — the LLM's low-level role *(new)*

**Why it exists.** Lexicons written by hand do not know this corpus's slang, sarcasm, ticker
shorthand or promo cadence; a general lexicon score on this data is weak. But asking an LLM to score
80,793 tweets is neither reproducible nor affordable. So the LLM does the part only it can do —
**read a sample and write down the vocabulary** — and deterministic code does the rest.

**Contract**

| | |
|---|---|
| Input | A capped, stratified sample: at most `induction.per_ticker` (default 60) tweets per ticker, drawn deterministically by (ticker, session) hash, plus percentage caps for promo, reply-shaped and cashtag-dense tweets, plus the §2.6 warnings and the currently frozen lexicon's terms |
| Output | `LexiconProposal` (strict JSON): `terms[]` each with `{term, family, polarity, weight, evidence_tweet_ids[], rationale}`, `relevance_rules[]`, `negation_patterns[]`, `sarcasm_markers[]`, and `abstain_reasons[]` |
| Budget | one call, capped tokens, temperature 0 where the provider supports it |
| Never | sees the full dataset; scores a row; assigns a status; writes a finding number |

**The validation gate (deterministic, and the reason this is safe).** The induction sample and the
validation slice are **disjoint** (split by session, not randomly). For every proposed term:

1. count its occurrences in the validation slice;
2. if occurrences < `induction.min_support` (default 10), **reject**;
3. else recompute its sign/weight from the validation slice and accept only if it agrees with the
   proposal (`induction.sign_agreement`, default ≥ 0.6) with a bootstrap CI that excludes zero;
4. the rest go to `artifacts/runs/<run_id>/rejected_proposals.json` with the failing rule named.

Only accepted terms enter the frozen lexicon. The agent's output is therefore never trusted
directly — it is a hypothesis list that must survive its own test.

**Frozen artifacts.** `artifacts/lexicons/<lexicon_id>.json`, where
`lexicon_id = sha256(canonical_json(accepted_terms) + induction_config + model_name + seed)[:12]`.
Every `ToolResult` and finding records the `lexicon_id` in its `data_version`, so a change in
vocabulary is visible in every downstream result.

**Shipped default.** `config/lexicons/default_v1.json` is committed and hand-written, so the no-key
run never depends on the induction stage existing. It bundles a VADER-compatible core plus a small
finance directional list (bullish/bearish), an uncertainty/hedging list, and attribution markers.

**Failure handling.** Invalid JSON → one repair attempt with the validation error → then fall back to
`default_v1.json` and log. Endpoint unreachable → fall back immediately. **The induction stage can
never fail a run.**

### 4.4 LLM client and configuration

`LLMClient.complete(messages, tools, response_schema) -> dict` is the only LLM touchpoint, shared by
the Induction, Explorer and Presenter agents.

- **Transport:** OpenAI-compatible `POST {base_url}/chat/completions` over the standard library
  (`urllib.request`), so **no vendor SDK is a dependency** and any compatible endpoint works
  (OpenAI, OpenRouter, vLLM, Ollama, LM Studio, a gateway). `requests`/`httpx` can be substituted
  behind the same interface.
- **Tool calling:** native `tools`/`tool_calls` when the endpoint advertises support; otherwise a
  bounded JSON-schema-in-prompt mode with the same validation. Selected automatically, recorded in
  the manifest.
- **Determinism:** `temperature=0`, fixed seed, and response caching keyed by
  `sha256(prompt + model + schema)`. A cached run is byte-identical and costs nothing.
- **Secrets:** never logged, never written to artifacts. The manifest stores `base_url` host, model
  name, sampling params, and a truncated prompt hash — never the key.

### 4.5 Sentiment family — 7 measures *(top-layer deduction)*

All seven are computed **deterministically** from the frozen lexicon and are per ticker-session.
Each is a deduction on top of the lower-layer text features, not a raw score dump.

| # | Measure | Definition | Conservative note |
|---|---|---|---|
| S1 | `sent_net` | mean over tweets of a signed polarity in [−1, 1] (lexicon weight sum + VADER compound, blended by config) | both components are offline and deterministic |
| S2 | `sent_pos_share` − `sent_neg_share` | share of tweets above/below the ±0.05 label band | the conventional VADER band, stated in every caption |
| S3 | `sent_intensity` | mean absolute polarity | separates "mildly positive, many tweets" from "strongly positive, few" |
| S4 | `sent_dispersion` | within-day std of polarity; plus `sent_agreement` = share on the majority side | the honest measure of *disagreement*; a day can be net-neutral and deeply split |
| S5 | `fin_directional` | (bullish-term count − bearish-term count) / total terms, using the induced finance list | lexicon-only; no model, no key |
| S6 | `sent_uncertainty` | share of tokens in the uncertainty/hedging list | hedging is a distinct signal from polarity, and it is cheap to compute |
| S7 | `sent_attribution` | share of polarity carried by *reported* speech (analyst/company/"according to") vs *asserted opinion* | a reported-cautious tweet is not a bullish author |

**Modifiers applied inside scoring, not reported as separate measures:** negation scope (a negator
within 3 tokens flips and damps the term weight) and a sarcasm-marker share reported as a caveat
metric, `sent_sarcasm_share`.

**Methodology statements that must accompany every sentiment result:** the polarity model name and
version · the `lexicon_id` · the label band used · `n_tweets` and `n_units` (ticker-days) ·
low-coverage and excluded-day counts · that VADER is general-purpose and weaker on financial slang
and sarcasm · that a finance-tuned transformer can sit behind the same `SentimentScorer` interface
later.

**Quality gates applied to the sentiment family, always reported, never silent:**
`min_tweets_per_day` (default 3, from the measured distribution) · `promo_excluded_share` ·
`own_text_only` (default **on** for tiers B and C, **off** for tier A so genuine shared market
chatter is kept) · `low_coverage_flag`.

### 4.6 Attention family — 7 measures *(top-layer deduction)*

Attention is a separate family because it is derivable without any lexicon, and because the profile
showed raw tweet count is **not** ticker attention.

| # | Measure | Definition |
|---|---|---|
| A1 | `n_tweets` | raw rows for ticker-session |
| A2 | `n_texts_unique` | distinct normalized texts — removes exact reposts |
| A3 | `n_texts_own` | texts not shared with any other ticker — **the ticker-specific attention measure** |
| A4 | `n_texts_shared` | texts also carried by other tickers — **market-wide attention spillover** |
| A5 | `own_text_share` | A3 / A1 — the tier detector, and a headline number for every ticker |
| A6 | `attn_velocity` | session-over-session change in A3, as a ratio |
| A7 | `attn_z` | **past-only** robust z-score (median/MAD) of A3 against its trailing 20-session baseline, shifted so day d sees only days before d |

Also computed and reported as supporting columns, not as headline measures: `promo_share`,
`reply_share` (rows starting with `@`), `cashtag_share`, `share_of_all_tweets`, and a
`spike_sustained` flag (spike persisting ≥2 sessions) versus a one-day burst.

**Why A3/A4 are the point.** AMZN, MSFT, PG, F and NOC have 100% shared text; their A1 looks large and
their A3 is near zero. Reporting only A1 would present a scraping artifact as market interest. The
attention family therefore always reports the header
`specific attention (A3) vs market-wide (A4)` and never A1 alone.

### 4.7 Price and quality features

- Price: `ret_contemp`, `abs_ret`, `fwd_ret_{1,2,5}`, `rv_5`, `rv_20`, `volume_z`.
- Quality: `low_coverage` (n_tweets below the configured minimum), `excluded_from_tests` with a
  reason, `tier`.
- All windows, thresholds and horizons live in `config/experiments.yaml`.

### 4.8 EDA tool library

Every analysis is a registered Python function with a Pydantic parameter schema and a uniform result.

```python
class ToolResult(BaseModel):
    result_id: str          # sha256(tool, canonical params, data_version)[:12] -> reproducible
    tool: str
    params: dict
    period: str             # "discovery" | "holdout" | "full"
    n_obs: int              # observations used
    n_units: int            # independent units (ticker-days) — the honest sample size
    n_dropped: int          # rows removed by this tool, with reasons in caveats
    stats: dict             # small JSON-safe numbers agents may cite
    table: list[dict] | None  # capped at 50 rows
    chart: ChartSpec | None   # declarative, renderer-agnostic (see §4.9)
    caveats: list[str]
    data_version: str         # data hash + config hash + lexicon_id + code version
```

**Tool catalog** (sentiment + attention + the descriptive layer around them):

| Group | Tool | Returns | Chart |
|---|---|---|---|
| Univariate | `coverage_by_ticker` | tweets, own-text tweets, trading days, tier per ticker | bar |
| | `distribution(column, by?)` | quantiles, skew, outlier share | histogram / box |
| | `text_stats` | length, cashtag/hashtag/URL share, promo share, fan-out share, top terms per sentiment label | bar |
| Temporal | `timeseries(metric, ticker\|pooled)` | series with rolling mean | line |
| | `attention_vs_price(ticker)` | A3/A4 tweet counts against close | dual-axis line |
| | `sentiment_vs_price(ticker)` | sentiment series against close | dual-axis line |
| Bivariate | `correlate(x, y, lag, ticker\|pooled)` | Spearman ρ, block-bootstrap CI, `n_units` | scatter |
| Conditional | `bucket_forward_returns(feature, buckets, horizon, ticker\|pooled)` | mean forward return per bucket vs unconditional, bootstrap CI | bar + error bars |
| Multivariate | `corr_matrix(metrics, level)` | correlation matrix | heatmap |
| Anomaly | `spike_days(metric, z_threshold)` | days beyond a robust z threshold | line + markers |
| | `sample_tweets(ticker, date, n≤10, strata)` | a few raw tweets for qualitative grounding | table |
| Validation | `holdout_check(result_id)` | re-runs the same call on the holdout period | paired bars |
| | `adjust_pvalues(result_ids)` | Benjamini-Hochberg adjusted q-values for a batch | table |

**Statistical defaults** (all in config):

- CIs use a **block bootstrap** (block length 5 sessions, 2000 resamples, fixed seed) because
  forward returns at h>1 overlap and daily series are autocorrelated.
- Pooled results use within-ticker standardized features and are **always shown next to per-ticker
  results and next to a TSLA-excluded pooled variant** (because TSLA is 46.3% of the corpus).
- Whenever a batch of ticker × horizon tests is screened, BH-adjusted q-values are reported.
- Chronological 70/30 discovery/holdout. No random splits, enforced in code.
- Discovery/holdout discipline is enforced **in code, not in prompts**: the Explorer's tools accept
  only `period=discovery`; the coordinator, never the agent, triggers `holdout_check`.

### 4.9 ChartSpec and the renderer

Tools do **not** build figures. They return a declarative, JSON-safe `ChartSpec`:

```python
class ChartSpec(BaseModel):
    kind: str            # "line" | "bar" | "bar_error" | "hist" | "scatter" | "heatmap" | "box" | "dual_line"
    title: str
    x: dict              # {label, values}
    series: list[dict]   # [{label, values, kind?}]
    annotations: list[dict] = []
    source_result_id: str
    caption: str         # always states period, n_units, and the data version
```

Rendering is a separate, swappable step with two backends:

- **PNG** — matplotlib with the `Agg` backend, written to `artifacts/runs/<run_id>/figures/`
  and opened in the OS default viewer. matplotlib is already available; no browser UI.
- **ASCII** — a compact sparkline/bar fallback drawn with `rich` for a quick in-terminal look when
  the user does not want a window.

Keeping specs and rendering apart is what makes the tool library unit-testable without a display and
lets the chart set change without touching any analysis.

### 4.10 Baseline playbook and agents

**Baseline playbook (no LLM).** Runs first on every run; roughly a dozen fixed tool calls: ticker
coverage and tiers; sentiment and return distributions; text statistics including promo and fan-out
shares; pooled attention (A3 vs A4) and sentiment time series; the correlation matrix over
sentiment, attention, return and volatility features; sentiment vs forward return (correlation plus
tercile buckets at h = 1, 2, 5); attention vs next-day absolute return and volatility; volume spike
days. This guarantees a complete report even if every agent call fails.

**Explorer agent (LLM).** Input: `profile.json`, the quality warnings, and a compact digest of
playbook results (stats and captions, not raw rows). Capabilities: the tool registry as
function-calling schemas; a budget of ≤12 tool calls; one retry on a schema error. Job: notice what
stands out, call follow-up tools (drill into one ticker, inspect a spike day with `sample_tweets`,
compare tickers), and return `FindingDraft` objects. Must not: state a number it did not get from a
tool, claim causation, or call anything outside the registry.

**Presenter agent (LLM).** Input: validated findings plus the stats of the results they cite. Output:
a short executive summary, one narrative paragraph per finding, a "what to look at next" list, and
chart captions. Every sentence containing a number carries a `[R:<result_id>]` reference. Style:
what was observed, how strong, how uncertain, what it does **not** show.

**No-key mode.** `NullAgent` selects findings from the playbook by simple rules (largest absolute
effects, biggest spikes) and fills templated text. The terminal and the report state plainly that
LLM features are off, and name the frozen `lexicon_id` in use.

**Claim checker (deterministic, not an LLM).** Rejects or downgrades agent output when: JSON fails
schema validation; a cited `result_id` does not exist or does not belong to this run; a number in the
claim text matches no value in the cited stats within rounding tolerance; or causal/advisory language
appears ("causes", "drives", "buy", "profitable", "will rise"). Rejected items are logged, counted
in the manifest, and displayed as unverified or dropped — never silently accepted.

### 4.11 Findings schema and status rules

Simplified from the full PRD insight schema so findings can later be upgraded into registry insights.

```json
{
  "schema_version": "0.2",
  "finding_id": "F-003",
  "title": "Specific tweet attention and next-day absolute return",
  "scope": {"tickers": ["pooled_ex_TSLA"], "period": "discovery", "tier": "A"},
  "claim_text": "…agent-written, numbers must match cited stats…",
  "evidence": [{"result_id": "r_9f2c…", "role": "primary"},
               {"result_id": "r_41ab…", "role": "holdout"}],
  "effect": {"metric": "spearman_rho", "value": null, "ci_95": null,
             "n_units": null, "q_value": null},
  "status": "observed",
  "limitations": [],
  "alternative_explanations": [],
  "origin": "explorer_agent",
  "run_id": "…",
  "lexicon_id": "…"
}
```

Unknown values stay `null` until a tool computes them. Nothing is filled in with plausible-looking
values.

**Status is assigned by rules, not by the LLM:**

| Status | Rule |
|---|---|
| `observed` | descriptive result with no hypothesis test (a trend, a distribution, a spike) |
| `supported` | tested; BH-adjusted q below alpha; CI excludes zero; same sign on the holdout period |
| `not_replicated` | passed on discovery but failed or flipped on the holdout |
| `inconclusive` | tested, but none of the above |

`supported` means "passed this demo's checks" and nothing stronger. It never implies a trading edge.
Every ticker-C finding, and every tier-B finding whose text is not ticker-specific, is forced to
`observed` at most.

### 4.12 Presentation — terminal first

**No separate UI.** Streamlit and Plotly are removed from the project.

At the end of a run the CLI prints, in the terminal:

1. **Overview** — dataset size, window, split dates, rows kept/flagged, tier table, promo and
   fan-out shares, and the §2.4 limits, in a compact `rich` layout.
2. **Summary** — the executive summary and each finding as a short block with its status badge,
   `n_units`, effect and CI, limitations, and the `result_id` of every number.
3. **The full Markdown report** is written to `artifacts/runs/<run_id>/report.md` (and
   `report.html`, self-contained with embedded PNGs) and its path is printed.

**Badges distinguish three things everywhere:** `[OBSERVED DATA]` · `[STATISTICAL ESTIMATE]` ·
`[AGENT INTERPRETATION]`. Wording is neutral ("largest observed change"), never trading advice.

**Chart menu.** After the report, the CLI lists the charts that this run actually produced, numbered,
each with its title, source `result_id` and `n_units`:

```
Available charts
 1  Tweets per ticker, specific vs market-wide        [bar]      r_7c1a…  n_units=25
 2  TSLA specific attention vs close                  [dual_line] r_2f90…  n_units=252
 3  Sentiment tercile → fwd_ret_1 (pooled ex-TSLA)    [bar_error] r_9f2c…  n_units=1184
 ...
Enter a number, "3,5", "all", "ascii 2", "md", "log", or "q":
```

The user types a number; that chart renders as a PNG and opens. `ascii <n>` renders in-terminal
instead. `all` renders everything. This keeps every figure traceable to a tool result and keeps the
surface to one window.

---

## 5. Configuration, secrets and packaging

### 5.1 Where the LLM details go — decided

**Split by secrecy, not by convenience:**

| What | Where | Why |
|---|---|---|
| base URL, model name, API key, enable flag | **`.env`** (git-ignored) | the key must never be committed |
| provider name, tool-calling mode, timeouts, max tokens, temperature, agent budgets | `config/settings.yaml` | safe to review, diff and commit |
| sampling caps, thresholds, windows, horizons, split fraction, bootstrap, alpha, tier rules | `config/experiments.yaml` | the experiment surface |
| committed template with placeholders | `.env.example` | onboarding without leaking |

`.env.example` as shipped:

```dotenv
# Copy to .env and fill in. .env is git-ignored and is never written into artifacts.
EDA_LLM_ENABLED=false
EDA_LLM_BASE_URL=http://localhost:11434/v1
EDA_LLM_API_KEY=
EDA_LLM_MODEL=qwen2.5:7b-instruct
EDA_LLM_TIMEOUT_S=120
```

`config/settings.yaml` never contains a secret; it references the environment and holds the rest:

```yaml
llm:
  provider: openai_compatible      # only transport implemented in v0
  base_url_env: EDA_LLM_BASE_URL
  api_key_env: EDA_LLM_API_KEY
  model_env: EDA_LLM_MODEL
  enabled_env: EDA_LLM_ENABLED
  tool_calling: auto               # auto | native | json_prompt
  temperature: 0
  max_tokens: 2000
  cache: true
  cache_dir: artifacts/llm_cache
agents:
  induction: {per_ticker: 60, max_total: 1500, min_support: 10, sign_agreement: 0.6}
  explorer:  {max_tool_calls: 12, retries_on_schema_error: 1}
  presenter: {max_findings: 8}
sentiment:
  method: vader_plus_lexicon      # vader_plus_lexicon | lexicon_only
  label_band: 0.05
  lexicon: artifacts/lexicons/<lexicon_id>.json   # falls back to config/lexicons/default_v1.json
paths:
  raw_tweets: data/raw/stock_tweets.csv.zip
  raw_prices: data/raw/stock_yfinance_data.csv
  artifacts: artifacts/runs
timezone:
  market: America/New_York
  close_hour: 16
```

**Precedence:** process environment > `.env` > `config/settings.yaml` defaults. Resolution is
centralised in `src/eda_demo/config.py`, which is the only module that reads `.env`. A startup line
prints the resolved *non-secret* settings and which source won.

```yaml
# config/experiments.yaml (excerpt — starting defaults)
split: {discovery_fraction: 0.70}
attention: {baseline_sessions: 20, spike_z: 3.0, sustained_sessions: 2}
sentiment: {min_tweets_per_day: 3, own_text_only: {tier_a: false, tier_b: true, tier_c: true}}
returns: {horizons: [1, 2, 5], basis: adj_close}
bootstrap: {block_sessions: 5, resamples: 2000, seed: 20260912}
multiple_testing: {alpha: 0.05, method: benjamini_hochberg}
tiers: {specific_own_text_share: 0.80, mixed_own_text_share: 0.10}
```

Every run manifest embeds the fully resolved `settings.yaml` and `experiments.yaml`, so a result can
be reproduced from the manifest alone.

### 5.2 Packaging

- `requirements.txt` with pinned versions: `pandas`, `numpy`, `scipy`, `pydantic>=2`, `PyYAML`,
  `python-dotenv`, `vaderSentiment`, `matplotlib`, `rich`. (The LLM transport is stdlib.)
- `pyproject.toml` for metadata and tool config only; no build step required.
- `python -m eda_demo <command>` is the single entry point, so nothing needs installing into the venv
  as a console script.
- `eda doctor` verifies: Python version · venv active · dependencies importable · raw files present
  with matching hashes · `data/raw` writable check (**must be read-only in use**) · `.env`
  parseable · LLM reachability via a 1-token ping · and writes nothing.

---

## 6. Repository layout

`.venv/`, `.env`, and `data/raw/` are part of the layout, as requested.

```
agentic-eda-demo/                      # == the workspace root
├── README.md                          # setup, assumptions, limitations, how to run each phase
├── pyproject.toml
├── requirements.txt                   # pinned
├── .gitignore                         # .venv/, .env, artifacts/, __pycache__/
├── .env.example                       # committed template (never .env itself)
├── .env                               # LOCAL ONLY, git-ignored: base URL, model, key
├── .venv/                             # LOCAL ONLY, git-ignored
├── SPEC.md                            # this document
├── STEP0_PROFILE.md                   # verified dataset profile (Step 0 evidence)
├── config/
│   ├── settings.yaml                  # paths, timezone, LLM wiring, agent budgets
│   ├── experiments.yaml               # windows, thresholds, horizons, split, bootstrap, alpha, tiers
│   └── lexicons/
│       └── default_v1.json            # shipped no-key lexicon
├── data/
│   └── raw/                           # original files, READ-ONLY, never modified
│       ├── stock_tweets.csv.zip
│       └── stock_yfinance_data.csv
├── docs/
│   ├── quality_report.md              # generated by the profiler
│   └── feature_book.md                # what every feature means and its caveats
├── src/eda_demo/
│   ├── __main__.py                    # CLI dispatch: doctor|profile|induce|features|playbook|charts|run
│   ├── config.py                      # the only reader of .env; precedence resolution
│   ├── ingest/                        # loaders.py (zip-aware), profile.py, validate.py, hashing.py
│   ├── prep/                          # clean.py, align.py (sessions), promo.py, fanout.py
│   ├── features/                      # attention.py, sentiment.py, price.py, panel.py, lexicon.py
│   ├── agents/                        # llm.py, induction.py, explorer.py, presenter.py, null_agent.py
│   ├── tools/                         # registry.py, schema.py, chart.py, one module per tool group
│   ├── findings/                      # schema.py, checker.py, status_rules.py
│   ├── report/                        # build_report.py, render_png.py, render_ascii.py, chart_menu.py
│   └── pipeline.py                    # coordinator: run_all()
├── tests/                             # unit/, integration/
└── artifacts/
    ├── lexicons/<lexicon_id>.json     # frozen accepted vocabularies
    ├── llm_cache/                     # response cache keyed by prompt hash
    └── runs/<run_id>/                 # profile.json, results/, findings.json,
                                       # report.md, report.html, figures/, manifest.json,
                                       # rejected_proposals.json, logs/
```

---

## 7. Run flow, manifest and new data

### 7.1 Run flow

1. `doctor`: check environment and input hashes.
2. Load inputs read-only, hash them, profile; stop on blocking errors.
3. Build canonical tables and the daily panel; log every cleaning decision and row-count change.
4. Resolve the lexicon: frozen `lexicon_id` if present and config-compatible, else induction
   (if an LLM is configured), else `default_v1.json`.
5. Run the baseline playbook; store `ToolResult`s.
6. Explorer agent loop on the **discovery** period only (skipped in no-key mode).
7. Coordinator runs `holdout_check` and `adjust_pvalues` for testable findings.
8. Claim checker validates findings; status rules assign status.
9. Presenter agent writes the narrative; the claim checker verifies it again.
10. Write `findings.json`, `report.md`, `report.html`, figures and `manifest.json`; print the
    overview and the chart menu.

### 7.2 Manifest — one per run

Run ID · input hashes (container and member) · resolved `settings.yaml` and `experiments.yaml` ·
`lexicon_id` and its provenance (induced / default / cached) · sentiment method and version ·
LLM provider, base-URL host and model (never the key) · tool-calling mode · random seed · start and
end times · data-quality warnings · the ordered list of tool calls with `result_id`s and
`n_dropped` · rejected agent outputs with reasons · rejected lexicon proposals with the failing rule ·
artifact paths.

**Reproducibility contract:** the same data and config yield identical `result_id`s and identical
stats. Only LLM prose may vary. (`result_id` excludes prose and excludes rendered figure bytes.)

### 7.3 New information arriving *(new)*

The system must assume fresh tweets will arrive, so the update path is explicit and cheap:

1. **Append-only ingest.** New tweets are added; existing rows are never rewritten. The panel is
   extended, not recomputed with different rules.
2. **Deterministic re-scoring first.** New tweets are scored with the already-frozen
   `lexicon_id`. No LLM call. This is the default and the common case.
3. **Drift check.** After ingest, `lexicon_coverage` = share of non-stopword tokens not covered by
   the frozen lexicon (plus VADER's). If it exceeds `drift.max_uncovered_token_share`
   (default 0.15) on a minimum volume of new text, a **re-induction proposal** is generated.
4. **Re-induction is additive and versioned.** A new `lexicon_id` is produced; the old one stays
   valid. Results record which lexicon scored them, so a metric change is always attributable to
   either data or vocabulary — never ambiguous.
5. **History is preserved.** Sessions already analysed are not silently restated. A change in a
   previously published number is reported as a diff in the run manifest.
6. **Schema drift is blocking, not silent.** If an arriving file's columns differ from what was
   profiled in §2.2, the run stops with a diff rather than guessing.

v0 implements steps 1–3 fully and 4–6 as logged, checked behaviour on an appended fixture.

---

## 8. Analyses the demo should speak to

Research questions, not expected effects. Any of them may return "nothing there", and the demo must
present that outcome as clearly as a positive one. Scope is deliberately limited to the sentiment and
attention families.

| # | Question | Core method | Main caveats |
|---|---|---|---|
| A | Does sentiment relate to next-session returns? | `bucket_forward_returns` on `sent_net` terciles, h = 1, 2, 5, vs unconditional baseline; per-ticker for tier A only, plus pooled and pooled-ex-TSLA; holdout replication | lexicon noise; low-coverage days; overlapping horizons; ~250 sessions per ticker |
| B | Does **specific** attention precede higher volatility? | `attn_z` high vs normal days against next-day `abs_ret` and `rv_5`; A3 versus A4 compared; dependence on a few event days checked | news drives both attention and volatility; one regime; promo text inflates raw counts |
| C | Is market-wide attention (A4) different from specific attention (A3)? | correlate A3, A4 and `own_text_share` against return, volatility and volume; per-ticker coverage-normalized comparison | fan-out is a scraping artifact, not trader behaviour |
| D | Do sentiment extremes precede reversal or continuation? | top/bottom decile of `sent_net`, split by prior-day return | small samples in extreme buckets |
| E | Do attention spikes cluster across tickers on the same session? | `corr_matrix` of A3 across tickers; co-spike days | very uneven per-ticker counts; calendar-wide news days dominate |

Plus the always-on descriptive layer: distributions, trends over time, day-of-week effects,
correlation heatmap, text statistics, promo and fan-out structure.

### 8.1 Families suggested but deliberately not in v0

You asked whether there are others. These are real and derivable from this dataset, but each needs
either an author field or a second data source, so they are proposed rather than built:

| Candidate family | What it would measure | Why deferred |
|---|---|---|
| **Novelty** | share of text that is new information vs recycled/duplicated commentary | needs near-duplicate detection and a cross-day similarity index; adjacent to A2/A4, which already capture the cheap part |
| **Coordination** | near-simultaneous identical or templated posting across tickers | the dataset has no author column, so this can only be measured at text level, which is exactly the fan-out already covered by A4 |
| **Narrative concentration** | how concentrated chatter is in a few themes | needs topic modelling; adds a non-deterministic component the conservative rules currently avoid |

Recommendation: keep v0 at **sentiment + attention**, and revisit Novelty only after the panel and
tools are stable, since it is the only one of the three that needs no new data source.

---

## 9. Testing

- **Leakage and alignment (synthetic fixture):** after-close and weekend tweets land in the next
  session; `attn_z` and sentiment z-scores on day d use only days before d; forward returns match the
  intended horizon; the last `h` sessions have null `fwd_ret_h`, not a wrapped value.
- **Calculations** against hand-computed values: returns, rolling volatility, robust z-score (median/MAD),
  bucket means, block bootstrap CI, BH adjustment.
- **Fan-out and promo handling:** a synthetic text shared by three tickers is counted once in A2,
  three times in A1, zero times in A3; `own_text_share` matches; a promo row is flagged and reported,
  never silently dropped.
- **Tier rules:** a ticker with own-text share below 0.10 is assigned tier C, excluded from sentiment
  and attention inference, and its findings are capped at `observed`.
- **Claim checker:** a made-up number is rejected; a valid cited number passes; causal wording is
  flagged; a `result_id` from another run is rejected.
- **Induction gate:** a deliberately wrong proposed term (right word, wrong sign) is rejected on the
  validation slice with the failing rule recorded; a term below `min_support` is rejected; the
  inductor's output can never change a number without passing the gate.
- **Determinism:** identical inputs and config produce identical `result_id`s and stats, with the LLM
  cache warm and with the LLM disabled entirely.
- **No-key integration test:** a small fixture dataset runs end to end and produces a complete report.
- **Schema tests:** `ToolResult`, `ChartSpec`, `Finding`, `LexiconProposal` validation including
  null handling and rejection of extra keys.
- **Chart contracts:** every tool that declares a chart yields a `ChartSpec` that the PNG and ASCII
  renderers both accept.

---

## 10. Acceptance criteria

1. Raw files are loaded read-only (in-memory zip read, no extraction); a data quality report is
   produced.
2. The real schema is mapped to the canonical tables; discrepancies from draft 0.1 are documented
   in `STEP0_PROFILE.md`.
3. The **no-key** pipeline completes and produces a complete report and chart menu.
4. At least three of the analyses in §8 run, each stating variables, horizon, baseline, `n_units`
   and `n_dropped`.
5. Predictive-style results use forward returns, past-only features and a chronological holdout.
6. Every figure and finding links to a `result_id` and its parameters.
7. Agent output is schema-validated, and every cited number matches a tool result.
8. The terminal separates observed data, statistical estimates and agent interpretation, and contains
   no trading advice.
9. The LLM's low-level influence is limited to the induced lexicon, which passes the deterministic
   validation gate and is recorded by hash in every affected result.
10. New tweets can be appended and scored with the frozen lexicon without any LLM call.
11. Unit and integration tests pass; the README states setup, assumptions and limitations.

---

## 11. Build plan — 5 phases, human gate at each

Order matters: hard data first, deterministic middle, agents last. **The agent stops at every gate
and waits for your confirmation or feedback before starting the next phase.** Each phase ends by
writing `reviews/PHASE_<n>_REVIEW.md` — a short document with what was built, how to run it, the
evidence it works, what was assumed, and the exact decision you are being asked to approve.

| Phase | Deliverable | Exit condition | Your role at the gate |
|---|---|---|---|
| **0 · Scaffold + Step 0 profiler** | venv + `.gitignore` + `requirements.txt` + `.env.example`; `eda doctor`; zip-aware loader; `eda profile`; `profile.json`; `docs/quality_report.md`; blocking/warning logic | Real schema, timezone, coverage tiers and quirks documented; blocking issues found or ruled out; **`STEP0_PROFILE.md` reproduced by code, not by hand** | Review the quality report and confirm the schema and tier rules before anything downstream is trusted |
| **1 · LLM wiring + Feature Induction Agent** | `config.py` precedence; `.env` handling; `LLMClient`; `eda doctor` LLM ping; sampler; `induction.py`; the deterministic validation gate; `LexiconProposal` schema; `default_v1.json`; **`eda induce` live smoke test** | A live endpoint returns a proposal; the gate accepts/rejects with recorded reasons; the frozen `lexicon_id` is written; with no key the whole path falls back to `default_v1.json` | **You put base URL, model name and key in `.env`**, run the smoke test, and review the sample analysis + the accepted/rejected term list |
| **2 · Canonical tables + features** | `clean.py`, `align.py`, `promo.py`, `fanout.py`; tweets/prices/daily_panel; **sentiment family (7)**; **attention family (7)**; price features; `docs/feature_book.md` | Alignment and leakage tests pass; row counts reconcile; every feature has a definition and a caveat; determinism tests pass | Review the feature book: confirm each sentiment and attention measure means what you expect and that the caveats are honest |
| **3 · Tool library + playbook + terminal** | `tools/` registry and all catalog tools; `ChartSpec`; PNG and ASCII renderers; numbered chart menu; baseline playbook; terminal overview; `report.md` / `report.html` | Playbook runs end to end **with no LLM**; `ToolResult`s persisted; charts render from stored specs; every figure caption carries its `result_id` | Read the full no-LLM report in the terminal, pick charts from the menu, and confirm the presentation is what you want before agents are layered on |
| **4 · Findings, checker, agents, manifest** | `findings/` schema + checker + status rules; `NullAgent`; Explorer and Presenter; holdout + BH in the coordinator; manifest; `reviews/` | Agent findings pass the claim checker; failures degrade gracefully to the templated report; manifest reproduces the run | Review one end-to-end agent run: check the rejected-output log and confirm each finding traces from card to tool call to raw stats |

Phase 4 is the last required phase. The incremental/new-data path (§7.3) is implemented through
Phases 2–4 and demonstrated on an appended fixture inside Phase 4, so it needs no sixth gate.

---

## 12. Traceability to the full PRD

| Full PRD component | In this demo | Upgrade path |
|---|---|---|
| Ingestion, inventory, quality report | `profile.json` + `docs/quality_report.md` + coverage tiers | richer schema contracts, near-duplicate detection |
| Cleaning and canonical data | two tables plus panel, logged decisions, promo/fan-out flags | asset-mapping audit, more sources |
| Feature layer | induced lexicon + sentiment (7) + attention (7) + price | disagreement across multiple sentiment models, FinBERT behind `SentimentScorer` |
| EDA engine | tool library + playbook + `ChartSpec` | regime and change-point tools |
| Insight discovery | Explorer agent plus fixed playbook | template-driven screening at scale |
| Evidence validation | holdout check, BH adjustment, bootstrap CIs | walk-forward validation, robustness suite, economic validation |
| Insight registry | `findings.json` per run | versioned registry with relationships and history |
| Agents | Induction, Explorer, Presenter, claim checker | Critic as a first-class agent, specialist agents |
| Presentation | terminal overview, `report.md`/`report.html`, numbered chart menu | experiment history page, insight explorer, feedback capture |
| Run manifest and reproducibility | manifest per run, hashed lexicons, LLM cache | code revision and agent versioning checks |

`ToolResult`, `ChartSpec`, `LexiconProposal` and `Finding` are the contracts that make this upgrade
possible. If they stay stable, the registry, critic and extra data sources can be added without
redesigning the tools or the presentation.

---

## 13. Open decisions

| # | Decision | Recommendation | Needed by |
|---|---|---|---|
| 1 | LLM provider, base URL and model for the three agent calls | any OpenAI-compatible endpoint; the transport is stdlib, so this is a `.env` value and not a code decision | Phase 1 |
| 2 | Does the endpoint support native tool calling, or must the JSON-in-prompt path be used? | `auto` detection; the smoke test in Phase 1 reports which was chosen | Phase 1 |
| 3 | Sentiment method for v0 | `vader_plus_lexicon`: VADER offline + the induced finance list. FinBERT stays behind the same interface | Phase 1 |
| 4 | Is `own_text_only` on for tier **A** too? | default **off** for tier A (keeps shared market chatter, which is real signal), **on** for tiers B and C | Phase 2 |
| 5 | `min_tweets_per_day` | 3 — from the measured distribution (36% of ticker-days have <3 tweets); revisit after the Phase 2 feature book, since it decides each ticker's usable-session count | Phase 2 |
| 6 | Are TSLA-excluded pooled variants mandatory everywhere, or only where TSLA is the driver? | mandatory wherever "pooled" appears | Phase 3 |
| 7 | Default chart backend on your machine | PNG via the OS viewer, with `ascii <n>` always available | Phase 3 |
| 8 | Critic pass (a third agent proposing confounds) | defer past v0; keep behind `agents.critic.enabled: false` | Phase 4 |
| 9 | Is `report.html` a primary deliverable or a convenience export? | convenience export; `report.md` is primary | Phase 3 |

---

## 14. Deliverable conventions

- **Markdown is the durable format.** Terminal output is for reading now; `report.md` and `findings.json`
  are the artifacts that outlive the session.
- **Every number in prose carries its `result_id`.** No exceptions, including captions and the
  executive summary.
- **Every exclusion is counted and named.** Nothing is dropped silently anywhere in the pipeline.
- **Neutral wording only.** "Largest observed change", "the estimate is imprecise", "this does not
  show". No advice, no direction, no return promises.
- **Honest negatives ship.** A null result is a finding and gets a card like any other.

## 15. Code conventions for comments and explanations

As requested: short, straightforward explanations of systems and functions, kept next to the code.

- **Every module** starts with a docstring naming its purpose, its inputs, its outputs, and the one
  design decision it embodies — in particular *why the conservative choice was made*.
- **Every public function** gets a one-to-three line docstring: what it returns, what units or shape
  that is, and what it deliberately does not do (e.g. "does not shift for leakage; the caller must
  pass an already-shifted baseline").
- **Comments explain why, not what.** A line like `# shift(1) so day d only sees days before d` is
  required; `# increment counter` is not.
- **Every tunable constant** is defined once in `config/` and referenced, never duplicated in code;
  if a constant is unavoidably inline, a comment names where its value came from (e.g. "0.05 =
  VADER's conventional label band" or "3 = median-of-distribution floor from STEP0_PROFILE §2.6").
- **Every statistical procedure** carries a comment stating its assumption and its failure mode
  (e.g. "block bootstrap because h>1 forward returns overlap; block length 5 = one trading week").
- **Caveats live in code as data.** A tool returning a suspicious number adds to `caveats` rather
  than relying on a docstring nobody reads at runtime.
- **No unexplained magic values, no dead code, no commented-out blocks.**

---

*Draft 0.2 ends. Awaiting your review at the Phase 0 gate: confirm the schema, the tier rules, and
whether the split date and `min_tweets_per_day` in §13 look right. Phase 0 needs no LLM key.*
