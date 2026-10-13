# Progress — read this first in a new session

Single source of truth for **state**: which phase is done, what is next, and what a fresh
session must know before touching anything. Design lives in `SPEC.md`; decisions live in
`docs/adr/`; each phase gate has a document in `reviews/`.

**Last updated:** end of Phase 4 · **Status:** Phases 0–4 complete; the demo runs end to end with and
without an LLM. The four gate documents are awaiting the human's review.

---

## 1. Where we are

| Phase | Scope | Status | Gate document |
|---|---|---|---|
| **0** | Scaffold + Step-0 profiler: read-only ingestion, blocking/warning checks, quality report | ✅ complete | `reviews/PHASE_0_REVIEW.md` |
| **1** | LLM wiring + Feature Induction Agent + deterministic validation gate | ✅ complete, live-tested | `reviews/PHASE_1_REVIEW.md` |
| **2** | Canonical tables, session alignment, sentiment (7) + attention (7) feature families | ✅ complete | `reviews/PHASE_2_REVIEW.md` |
| **3** | EDA tool library, baseline playbook, terminal report + numbered chart menu | ✅ complete | `reviews/PHASE_3_REVIEW.md` |
| **4** | Findings, claim checker, status rules, Explorer + Presenter agents, manifest, append path | ✅ complete, live-tested | `reviews/PHASE_4_REVIEW.md` |

**Next action:** the human reads the four gate documents. Phase 4 is the last required phase; SPEC.md
§7.3 (new data) is implemented inside it, so there is no sixth gate.

**Open items the human may want to change** (all config, no code): the induction sample size
(`agents.induction.max_total`, strata caps), `min_tweets_per_day` and `own_text_only` (they cost 44% of
otherwise usable ticker-sessions — see `reviews/PHASE_2_REVIEW.md` §4), and `charts.default_backend`.

---

## 2. How to run what exists

From the repository root, always with the venv interpreter:

```powershell
.\.venv\Scripts\python.exe run.py doctor                    # environment, inputs, LLM reachability ping
.\.venv\Scripts\python.exe run.py profile                    # dataset profile -> profile.json + quality report
.\.venv\Scripts\python.exe run.py features                   # panel + docs/feature_book.md (cached)
.\.venv\Scripts\python.exe run.py induce                     # live induction + validation gate
.\.venv\Scripts\python.exe run.py playbook                   # the fixed no-LLM analyses + chart menu
.\.venv\Scripts\python.exe run.py run                        # the full pipeline (agents when a key is set)
.\.venv\Scripts\python.exe run.py run --no-agents            # full pipeline, templated narrative
.\.venv\Scripts\python.exe run.py charts --run-id <run_id>   # re-open the chart menu for a finished run
.\.venv\Scripts\python.exe run.py append --file <new.csv>    # new data: append-only, no LLM call
.\.venv\Scripts\python.exe -m pytest                         # 120 tests
```

Every command takes `--dataset dev|full`. `dev` is the fast 26-session slice; **`full` is required
before any number is reported.**

Outputs of `run` live in `artifacts/runs/<run_id>/`: `report.md` (primary), `report.html`,
`findings.json`, `results/results.json`, `manifest.json`, `figures/`, and `rejected_proposals.json`
when induction ran. The panel cache is `artifacts/cache/`; the frozen lexicons are
`artifacts/lexicons/`.

---

## 3. Environment facts a new session will trip over

These cost real time to rediscover. Nothing else in the repo depends on them.

1. **Never use `tempfile` in the workspace — not `TemporaryDirectory`, not `mkdtemp`.** The
   platform temp directory allows creating files but denies the `scandir`/`chmod` that teardown
   performs, and a `mkdtemp` directory created inside the workspace also becomes undeletable under
   the sandbox. Either way the test body succeeds and the failure appears during cleanup, which
   masks the real result. `tests/fixtures.py` provides `scratch()`, a **fixed-named** git-ignored
   directory under `.tmp-tests/`. Copy that pattern. `pyproject.toml` also disables pytest's cache
   plugin for the same reason. (`tests/test_phase0.py` carries its own copy of the same helper.)
2. **Creating `.venv` needed a one-time escalation.** `ensurepip` writes its bootstrap wheel to
   `%TEMP%`, which the sandbox denies. If the venv is ever rebuilt and `python -m venv .venv`
   fails with `PermissionError` on a temp path, that is this, not a broken Python.
3. **The console here is cp1252.** Printing Unicode (block characters, `—`, `→`) to a plain
   `print` raises `UnicodeEncodeError`. `report/render_ascii.py` therefore picks an ASCII ramp when
   `sys.stdout.encoding` cannot encode the blocks, and `chart_menu._PlainConsole` replaces rather
   than raises. Rich handles this itself, which is why the CLI looks fine.
4. **The 1:9 split files carry a UTF-8 BOM**, because `split_datasets.py` wrote them with
   `encoding="utf-8-sig"`. All reading goes through `utf-8-sig`, which is correct either way.
5. **`Close` in the price file is already split-adjusted.** Do not add split handling; it is
   asserted on every run via `split_return_limit`.
6. **pandas' default CSV float parser is not correctly rounded.** The panel cache is written with
   `float_format="%.17g"` and read with `float_precision="round_trip"`; without both, a cached run
   disagrees with a fresh one in the last bits of a float column, which reached a rank correlation
   in the fourth decimal. See ADR-0007.
7. **`itertuples` renames columns whose names start with an underscore** (`row._z` becomes `row._1`).
   Two bugs came from this; use plain identifiers for any column read through `itertuples`.
8. **A tool's own `period` parameter is the one that matters.** `ToolRegistry.call(..., period=…)`
   overwrites it, which is what makes `holdout_check` work. Never read `period` from anywhere else.

---

## 4. Which dataset is in use

Set in **`config/settings.yaml`** → `datasets.active`, or per run with `--dataset <name>`.

| Name | Files | Size | Use |
|---|---|---|---|
| `dev` *(active)* | `splits/tweets_validation.csv`, `splits/prices_validation.csv` | 8,079 tweets · **26 sessions** | correctness only |
| `full` | `stock_tweets.csv.zip` (member read in memory), `stock_yfinance_data.csv` | 80,793 tweets · 252 sessions | any real analysis |

**The `dev` slice cannot support conclusions** (26 sessions, no usable 20-session baseline) and it
yields only 149 scored ticker-sessions. Phases 2–4 numbers reported here come from `full`.

Manual ticker subsets live in **`config/experiments.yaml`** → `scope.include` / `scope.exclude`,
applied before any metric by the profiler *and* every later stage (one implementation, in
`prep/scope.py`).

---

## 5. Verified facts the design rests on

Full evidence in `STEP0_PROFILE.md`; reproduced by `run.py --dataset full profile` and again by the
Phase 2 panel. The Phase 2 numbers are in `reviews/PHASE_2_REVIEW.md` §3.

| Fact | Value |
|---|---|
| Tweets / prices | 80,793 rows · 6,300 rows = 25 tickers × 252 sessions |
| In window / past the last session | 80,758 / 35 (counted, flagged, excluded from the panel) |
| Window | 2021-09-30 → 2022-09-29, one regime |
| Cross-ticker fan-out | 6,939 cleaned texts under >1 ticker |
| Fully generic tickers | AMZN, MSFT, PG, F, NOC — 100% cross-posted (tier C) |
| TSLA share of corpus | 46.3% |
| Promo-shaped rows | 45,068 (55.8%) |
| Rolls a session | 51.7% (rows that actually move to a later session) |
| Coverage tiers | A=3 (TSM, TSLA, NIO) · B=17 · C=5 |
| Discovery / holdout | 2021-09-30 → 2022-06-10 (176 sessions) / 76 sessions |
| Sentiment coverage | 1,625 of 6,300 ticker-sessions scored (25.8%); 2,914 would pass the tweet minimum without the own-text gate |
| Strongest real result | specific attention ↔ same-day \|return\|, pooled ex-TSLA: rho ≈ 0.11, q ≈ 0.007, replicated on the holdout |
| Sentiment → forward return | inconclusive at h = 1, 2, 5 (q ≈ 0.76–0.96) |

---

## 6. What exists in the repo

```
run.py                       entry point (only file that touches sys.path)
config/settings.yaml         dataset selection, columns, market clock, LLM wiring, agent budgets
config/experiments.yaml      every window, threshold, horizon, seed and gate
config/lexicons/default_v1.json   the shipped no-key lexicon (162 terms, six families)
src/eda_demo/config.py       the only reader of .env and YAML
src/eda_demo/ingest/         loaders.py, validate.py, profile.py
src/eda_demo/prep/           clean.py, align.py, scope.py, build.py (cache-aware panel access)
src/eda_demo/features/       lexicon.py, sentiment.py, attention.py, price.py, panel.py,
                             definitions.py, feature_book.py
src/eda_demo/agents/         llm.py, sampler.py, induction.py, explorer.py, presenter.py, null_agent.py
src/eda_demo/tools/          schema.py, registry.py, descriptive.py, relational.py, validation.py,
                             playbook.py, common.py
src/eda_demo/stats/          inference.py (session-block bootstrap, Spearman, BH)
src/eda_demo/findings/       schema.py, checker.py, status_rules.py
src/eda_demo/report/         build_report.py, render_png.py, render_ascii.py, chart_menu.py
src/eda_demo/pipeline.py     the coordinator (stage functions + run_pipeline)
src/eda_demo/append.py       the new-data path
tests/                       test_phase0..4.py + fixtures.py (120 tests)
docs/                        PROGRESS.md, feature_book.md (generated), quality_report.md (generated), adr/
reviews/                     PHASE_0..4_REVIEW.md — one gate document per phase
artifacts/runs/<run_id>/     report.md/html, findings.json, results/, manifest.json, figures/
artifacts/lexicons/          frozen induced lexicons + active.json pointer
artifacts/cache/             panel + scored tweets keyed by data_version (fingerprint-checked)
```

---

## 7. Open decisions carried forward

| # | Decision | Needed by | Current default |
|---|---|---|---|
| 1 | LLM base URL, model, key | done — `.env` is populated and the live smoke test passes | `gpt-5.6-luna` on `api.experientiallabs.ai`, native tool calling |
| 2 | Keep the induced lexicon active? | any time | yes: `8ad5ddd16848` (164 terms) is active; `--lexicon default` reverts for one run |
| 3 | `min_tweets_per_day` / `own_text_only` | any time | 3 · on for tiers B and C — this is the gate that costs 44% of usable units |
| 4 | Chart backend default | any time | PNG via the OS viewer; `ascii <n>` always available |
| 5 | Induction sample size | any time | 1,375 tweets (84k-token prompt, ~$0.02); lower `max_total` to ~600 and `strata_caps.promo` to 0.10 for a cheaper call |

---

## 8. Working agreement

- One phase at a time; write `reviews/PHASE_<n>_REVIEW.md` and **stop**.
- Every code explanation stays short and states *why* the conservative choice was made (§15 of `SPEC.md`).
- Record any decision that a future session must not silently reverse as a new ADR in `docs/adr/`.
- Keep the project short. This is a demo of a pattern, not a platform.
- **The three rules that are easiest to undo by accident:** a tool result must be traceable
  (`result_id` + `data_version` on every number); the holdout is triggered by the coordinator and never
  by an agent; and no finding may exceed `observed` without a q-value, an excluding CI and a holdout
  sign.
