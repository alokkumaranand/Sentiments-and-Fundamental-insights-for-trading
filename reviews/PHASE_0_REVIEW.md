# Phase 0 review — scaffold + Step-0 profiler

> **Confirmed.** The human approved proceeding, and Phases 1–4 were built on this scaffold. The
> decisions in §6 are unchanged; §3's evidence was reproduced independently by the Phase 2 panel
> (see `reviews/PHASE_2_REVIEW.md` §3). Kept as written, because the gate document is also the
> record of what was true at the time.

**Gate: approved.** Phase 1 was built on top of it.

Phase 0 had to end with "the real schema, timezone, coverage and quirks documented, and blocking
issues found or ruled out". That is done, and — more usefully — the profiler reproduces by code
the same numbers I first derived by hand, which is the main evidence that it is correct.

---

## 1. What was built

| File | Purpose |
|---|---|
| `run.py` | entry point; the only file that touches `sys.path` (no install step, no console script) |
| `src/eda_demo/config.py` | the only reader of `.env` and the YAML configs; precedence lives here |
| `src/eda_demo/ingest/loaders.py` | read-only CSV and zip-member loading, SHA-256 hashing, no extraction |
| `src/eda_demo/ingest/validate.py` | the five blocking checks and the warning collector |
| `src/eda_demo/ingest/profile.py` | profiling metrics + the Markdown quality report |
| `src/eda_demo/__main__.py` | CLI: `doctor`, `profile` |
| `tests/test_phase0.py` | 17 tests on synthetic fixtures |
| `config/settings.yaml` | **dataset selection** (commented), column mapping, market clock, LLM wiring |
| `config/experiments.yaml` | every window, threshold, horizon and seed in one place |
| `README.md`, `docs/PROGRESS.md`, `docs/adr/*`, `requirements.txt`, `.env.example`, `.gitignore` | setup, state, decisions, environment |

Nothing was modified in the raw inputs. The tweets zip is read **in memory** and never extracted.

## 2. How to verify it yourself

```powershell
.\.venv\Scripts\python.exe run.py doctor
.\.venv\Scripts\python.exe run.py profile                    # dev slice
.\.venv\Scripts\python.exe run.py --dataset full profile      # whole corpus
.\.venv\Scripts\python.exe -m pytest
```

Expected: `doctor` reports all required checks passed; `profile` prints the overview and writes
`artifacts/runs/<run_id>/profile.json` + `quality_report.md`; `pytest` reports **17 passed**.

## 3. Evidence

**Tests:** 17 passed under both `pytest` and `python -m unittest discover` (~0.8 s). They cover
read-only loading, BOM handling, zip-member reading and non-extraction, each blocking check, the
fan-out/own-text calculation, the after-close and non-trading-date buckets, the ragged-session
warning, the split check, tier thresholds from config, and JSON-serialisability + reproducibility
of the profile. Fixtures are synthetic and tiny, so the suite stays valid if the data is swapped.

**Determinism:** two consecutive `profile` runs on `dev` produced the same `data_version`
(`74f5357aabe1`).

**Code reproduces the hand analysis.** A run on `full` produced exactly the figures I derived
independently before writing any code:

| Metric | Hand | Profiler |
|---|---|---|
| tweet rows / distinct texts | 80,793 / 64,479 | 80,793 / 64,479 |
| rows in cross-posted text | 28.5% | 23,060 (28.5%) |
| promo-shaped | 55.8% | 45,069 (55.8%) |
| non-trading dates | 22.1% | 17,866 (22.1%) |
| `Adj Close == Close` | 3,539 | 3,539 |
| split check (max 1-day move) | 0.351 NFLX | 0.351 NFLX, passed |
| coverage tiers | A=3 B=17 C=5 | A=3 B=17 C=5 |
| ragged sessions | 0 | 0 |
| HTML-entity rows | 7,745 | 7,745 |

## 4. What the profiler found

**No blocking errors.** Schema, tickers, date range and price completeness all match what
`SPEC.md` now assumes. Three findings drove design changes, all already reflected in `SPEC.md`:

1. **Cross-ticker fan-out.** 6,948 texts appear under more than one ticker, covering 28.5% of rows.
   AMZN/MSFT/PG/F/NOC are **100%** cross-posted — their corpora are not their own. This is why
   coverage tiers and the specific-vs-market-wide attention pair exist.
2. **Only 26 sessions in the `dev` slice, with a ragged final session.** The 1:9 split cut
   mid-session, so `2021-11-04` exists for 5 of 25 tickers. The profiler catches this as a warning
   (not blocking: an incomplete session is *uncertain*, not *wrong*).
3. **The dev slice has 2 tier-A tickers, not 3.** NIO's own-text share is 0.78 over this shorter
   window versus 0.85 on the full corpus. Tier assignment is window-dependent — so tiers must only
   be compared within one dataset slice. This is recorded in ADR-0001.

## 5. Assumptions I made — please correct me if any is wrong

1. **"Use the test one" = the small `*_validation.csv` files.** Your split script's default is
   chronological with the earliest 10% as `validation`, so `dev` points at those two files. The
   `train` files are left completely untouched.
2. **`dev` stays the default dataset** for phases 1–3 so building stays fast, with `full` used
   before any number is reported. Phase 2 will need `full` to exercise a 20-session window at all.
3. **Blocking vs warning split is narrow.** A check is blocking only when continuing would produce
   a *wrong* answer rather than an *uncertain* one. A ragged session, thin coverage and heavy promo
   share are therefore warnings carried into the report, never silent drops.
4. **The split-continuity check is blocking.** An unadjusted split would corrupt every return, so
   the run stops. `--no-check-splits` downgrades it to a note if you ever need to inspect such data.
5. **Splitting is not re-done.** I left your split as generated rather than re-cutting it on whole
   sessions, since you asked to keep things simple and untouched.

## 6. Decisions at this gate

After your feedback, everything here is decided except the one thing only you can supply.

| # | Item | Resolution |
|---|---|---|
| 1 | Tier thresholds (A ≥ 0.80, B 0.10–0.80, C < 0.10 own-text share) | **Kept as default**, commented in `config/experiments.yaml`. They are not a preference: 5 of 25 tickers have *zero* text of their own, so without this their per-ticker numbers are wrong rather than merely noisy. Change freely. |
| 2 | `min_tweets_per_day` | **Demoted.** A commented default (3) you may change at will; no longer a decision brought to you. It is not a dataset selector — it decides which ticker-days have enough text for a sentiment score to mean anything. |
| 3 | `dev` vs `full` | **Decided:** `dev` is the default for building (fast), `full` is required before anything is reported. Recorded in ADR-0001. |
| 4 | Ragged final session | **Decided:** dropped for cross-sectional work, flagged in the overview. It is 1 session of 26 in the `dev` slice; the full corpus has 0. |
| 5 | Manual ticker subset | **Added at your request.** `config/experiments.yaml` → `scope.include` / `scope.exclude`, commented, applied before any metric. Measured on `dev` with the 5 generic tickers excluded: 865 rows dropped, 20 tickers left, tier C = 0, shared-text rows fall 1,831 → 901. |
| 6 | **LLM endpoint** (`EDA_LLM_BASE_URL`, `EDA_LLM_MODEL`, `EDA_LLM_API_KEY` in `.env`) | **Needs you**, and only for the live smoke test at the end of Phase 1. Phase 1's code and Phases 0–3 as a whole run without it. |

To supply it: `Copy-Item .env.example .env`, then set `EDA_LLM_ENABLED=true` and fill in the URL,
model and (if remote) key. Any OpenAI-compatible endpoint works; a local server needs no key.

## 7. What Phase 1 will do

- `config.py` precedence + `.env` loading, verified end to end.
- `LLMClient` on the standard library (`urllib`) against any OpenAI-compatible
  `/chat/completions`; native tool calling with an automatic JSON-in-prompt fallback; response cache.
- `eda doctor` gains a 1-token reachability ping that never logs the key.
- The **Feature Induction Agent**: capped stratified sampler → `LexiconProposal` → the
  deterministic validation gate on a session-disjoint slice → frozen, hashed lexicon.
- The shipped no-key `config/lexicons/default_v1.json`.
- A live smoke test you can run once your endpoint is in `.env`.

**What I will need from you then:** the base URL, model name and (if remote) an API key in `.env`.
Nothing is needed for that until Phase 1 starts.
