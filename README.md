# Agentic EDA Demo

A small Python tool that takes **stock tweets + daily stock prices** and produces an exploratory
data analysis: a written report, charts, and a machine-readable list of findings.

The one idea worth remembering: **code does the computing, the AI only does the talking.** Every
number in the report comes from a checked calculation. The AI never invents a figure, and the whole
pipeline still runs with no AI key at all.

> Status: Phases 0–4 complete. It runs end to end with or without an LLM endpoint.

---

## Index

- [What is this?](#what-is-this)
- [What you need first](#what-you-need-first)
- [Quick start (5 minutes)](#quick-start-5-minutes)
- [The commands](#the-commands)
- [Two mistakes every new user makes](#two-mistakes-every-new-user-makes)
- [Getting a real analysis (not a test run)](#getting-a-real-analysis-not-a-test-run)
- [Choosing or changing the dataset](#choosing-or-changing-the-dataset)
- [Where the results go](#where-the-results-go)
- [How it works, in one screen](#how-it-works-in-one-screen)
- [Running it with or without AI](#running-it-with-or-without-ai)
- [What the data is, and its limits](#what-the-data-is-and-its-limits)
- [Project layout](#project-layout)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Where to read more](#where-to-read-more)

---

## What is this?

Input: two files — tweets that mention stock tickers, and daily prices for those tickers.

Output, in `artifacts/runs/<run_id>/`:

- `report.md` / `report.html` — the readable analysis, with numbers and charts
- `findings.json` — the findings, each one tied to the calculation that produced it
- `manifest.json` — exactly which data and settings produced this run

It is a **demo of a pattern**, not a trading system. Nothing it outputs is trading advice.

---

## What you need first

| Requirement | Notes |
|---|---|
| Python 3.11+ | Verified on 3.13.6 |
| The two data files | Already in the repo: `stock_tweets.csv.zip` and `stock_yfinance_data.csv` |
| An LLM endpoint *(optional)* | Only needed for the two AI layers. See [Running it with or without AI](#running-it-with-or-without-ai) |
| Windows / PowerShell | All examples below use PowerShell; on macOS/Linux use `./.venv/bin/python` instead |

---

## Quick start (5 minutes)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env        # optional, only if you want the AI layers

.\.venv\Scripts\python.exe run.py doctor     # checks everything, changes nothing
.\.venv\Scripts\python.exe run.py profile    # look at the data quality report first
.\.venv\Scripts\python.exe run.py --dataset full run
```

That last line is the one that produces the actual analysis. See
[Getting a real analysis](#getting-a-real-analysis-not-a-test-run) for why `--dataset full` matters.

---

## The commands

`run.py` is the only entry point. It takes **one command**, and every command also accepts
`--dataset dev|full`.

| Command | What it does | Uses AI? |
|---|---|---|
| `doctor` | Checks Python, packages, data files and the AI endpoint. Writes nothing. | No |
| `profile` | Data quality report → `profile.json` + `docs/quality_report.md` | No |
| `features` | Builds the ticker-day table and `docs/feature_book.md` | No |
| `induce` | Asks the AI for a sentiment vocabulary, then a strict check accepts or rejects it | 1 call |
| `playbook` | The fixed analyses + report + chart menu | No |
| `run` | **Everything**: playbook + AI follow-ups + holdout test + findings + report | Yes* |
| `charts` | Re-opens the numbered chart menu for a finished run | No |
| `append` | Adds new tweets to the existing data and checks for drift | No |

\* `run` still works without a key — it just writes a templated narrative instead of an AI-written
one. You can force that with `run --no-agents`.

Useful extras:

```powershell
.\.venv\Scripts\python.exe run.py -h                        # list all commands
.\.venv\Scripts\python.exe run.py run -h                    # options for one command
.\.venv\Scripts\python.exe run.py run --no-agents           # skip the AI narrative
.\.venv\Scripts\python.exe run.py charts --run-id <run_id>  # reopen charts from an old run
.\.venv\Scripts\python.exe run.py append --file new.csv     # fold in newly arrived tweets
```

Charts are shown through a small numbered menu after a run: type `3`, `3,5`, `all`, `ascii 2`,
`md`, `log`, or `q`.

---

## Two mistakes every new user makes

**1. Running `run.py` with no command.**

```
eda_demo: error: the following arguments are required: command
```

This is not a bug. `run.py` needs a command from the table above. Add one:

```powershell
.\.venv\Scripts\python.exe run.py doctor
```

**2. Believing the default `dev` results.**

The default dataset is `dev` — a deliberately tiny slice (26 trading sessions) that exists only to
check the code runs correctly. **No real conclusion can be drawn from it.** For anything meaningful
you must pass `--dataset full`.

---

## Getting a real analysis (not a test run)

```powershell
.\.venv\Scripts\python.exe run.py --dataset full run
```

`full` is the real corpus: about 80,000 tweets and one year of prices. This is the command that
answers "what does this dataset actually say?"

A sensible order the first time, because it fails early and cheaply if something is wrong:

```powershell
.\.venv\Scripts\python.exe run.py doctor                  # 1. is the environment sane?
.\.venv\Scripts\python.exe run.py profile                 # 2. is the data sane? read quality_report.md
.\.venv\Scripts\python.exe run.py --dataset full induce   # 3. freeze the vocabulary (optional)
.\.venv\Scripts\python.exe run.py --dataset full run      # 4. the actual analysis
```

Runs are cached, so repeating a step you have already done is cheap.

---

## Choosing or changing the dataset

There are three separate knobs. Most people only ever need the first.

**1. Which dataset runs** — `config/settings.yaml`, line 19:

```yaml
datasets:
  active: dev        # change to: full
```

Or override it for a single run with `--dataset full`, which changes nothing on disk.

**2. Bringing your own files** — still `config/settings.yaml`. Add an entry under
`datasets.sources:` with the paths to your tweets and prices file, then set `active:` to its name.
No code changes.

```yaml
    mydata:
      tweets: my_tweets.csv
      prices: my_prices.csv
      tweets_member: null      # null for a plain CSV; the member name if it is a ZIP
```

**3. Renaming columns** — same file, the `columns:` block. If your CSV uses different headers,
map them there instead of editing any Python.

**4. Narrowing the data by ticker** — `config/experiments.yaml`, lines 18–20. This filter is applied
before every calculation, so the reported counts always match what was actually used:

```yaml
scope:
  include: null                          # e.g. [TSLA, TSM, NIO]
  exclude: []                            # e.g. [AMZN, MSFT, PG, F, NOC]
```

**5. Tuning the analysis** — `config/experiments.yaml` is the single home for every window,
threshold, horizon, bootstrap count and significance level. Change them there; nothing is
hard-coded elsewhere.

---

## Where the results go

```
artifacts/runs/<run_id>/      report.md, report.html, findings.json,
                              results/results.json, manifest.json, figures/
docs/quality_report.md        the latest data quality report
docs/feature_book.md          what every feature means
artifacts/cache/              cached tables, keyed to the data version
artifacts/lexicons/           frozen vocabularies and the active one
artifacts/llm_cache/          cached AI answers, so a repeat run costs nothing
```

Deeper reference: `docs/adr/0001-dataset-slices-and-paths.md`.

---

## How it works, in one screen

```
two input files (never modified; ZIPs are read in memory)
  -> clean tweets + prices, aligned to the 16:00 ET market close      no AI
  -> sentiment vocabulary: shipped, or AI-proposed and strictly checked   1 AI call
  -> one row per ticker per day: sentiment features + attention features  no AI
  -> 14 calculation tools, a fixed playbook, chart specs              no AI
  -> AI picks a few follow-up questions to investigate (max 12)       AI
  -> holdout test + correction for multiple comparisons               no AI
  -> every claim checked against the tool result it cites             no AI
  -> AI writes the prose, re-checked arithmetically                   AI
  -> report.md / report.html / findings.json / manifest.json + charts
```

The AI has exactly two narrow jobs: **propose vocabulary** (once, and a strict check on separate
data accepts or rejects each term) and **explain results** (with every number it writes verified
against its source). It never scores a row, never sees the whole dataset, and never decides whether
a finding is supported.

---

## Running it with or without AI

**Without a key** — everything works. The shipped vocabulary
(`config/lexicons/default_v1.json`) is used and the report gets a templated narrative. This is the
intended default and the honest way to see the deterministic layer.

**With a key** — copy `.env.example` to `.env` and set:

```ini
EDA_LLM_ENABLED=true
EDA_LLM_BASE_URL=http://localhost:11434/v1     # any OpenAI-compatible server
EDA_LLM_API_KEY=                                # empty for a local server
EDA_LLM_MODEL=qwen2.5:7b-instruct               # whatever your server expects
```

Settings priority is: environment variables → `.env` → `config/settings.yaml`. The API key is never
logged and never written into `artifacts/`.

**Worth knowing:** the AI's *choices* are not reproducible, even at temperature 0. The calculations
are. So two runs may investigate slightly different questions, but every number they report is
traceable to the tool result it cites.

---

## What the data is, and its limits

One year: 2021-09-30 → 2022-09-29, which includes the 2022 drawdown. Nothing here generalises
beyond that window. It is a demo dataset, so read these before quoting any number:

- **The unit is a ticker-day, not a tweet.** 80,793 tweets is not 80,793 independent observations.
  Every statistic reports `n_units` next to `n_obs`.
- **28.5% of tweets are cross-posted** — the same text appears under several tickers. Five tickers
  (AMZN, MSFT, PG, F, NOC) are 100% cross-posted, so a score for them measures generic chatter, not
  the company.
- **TSLA is 46.3% of the corpus.** Pooled results are always shown beside a TSLA-excluded version.
- **55.8% of rows are promotional in shape**, and many tweets land after the close, so they count
  toward the next session.
- **Only ~26% of ticker-days get a sentiment score.** The rest have too few tweets or only shared
  text, and the excluded count is printed wherever it matters.
- **There is no author field**, so bot detection and unique-user metrics are impossible.
- Prices are already split-adjusted — the tool verifies this on every run and refuses to continue
  if it sees a suspicious jump.

`supported` on a finding means "passed this demo's checks" and nothing stronger.

---

## Project layout

```
run.py                  the entry point — run this, nothing else
config/                 settings.yaml, experiments.yaml, lexicons/
src/eda_demo/           the package: ingest, prep, features, agents, tools, stats,
                        findings, report, pipeline.py
tests/                  ~120 tests on synthetic fixtures
docs/                   PROGRESS.md, feature_book.md, quality_report.md, adr/
reviews/                one gate document per completed phase
artifacts/              outputs, caches and frozen vocabularies
```

The code lives under `src/` and is not pip-installed. `run.py` is simply the one line that puts
`src/` on the import path, which is why you run `run.py` rather than `python -m eda_demo`.

---

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Around 120 tests, all on small synthetic fixtures — no network, no AI, and they do not touch your
real data.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error: the following arguments are required: command` | You ran `run.py` with no command. Add one, e.g. `run.py doctor`. |
| `ModuleNotFoundError: pandas` | The venv is not set up, or you used the system Python. Always use `.\.venv\Scripts\python.exe`. |
| `doctor` says "not running inside .venv" | Same as above — use the venv interpreter. |
| `doctor` says `llm ping: failed` | The endpoint is unreachable or the model name is wrong. Not fatal: runs still work without AI. |
| The report looks thin | You are probably on the default `dev` slice. Add `--dataset full`. |
| `UnicodeEncodeError` on a plain `print` | The Windows console is cp1252. Use the CLI, which handles this itself. |
| Statistical conclusion looks odd | Check `docs/quality_report.md` first — the data may have excluded most of what you expected. |

If a run fails on data checks, the error message names the file and the offending column. That is
usually a `columns:` mapping that does not match your CSV.

---

## Where to read more

| Document | What is in it |
|---|---|
| `docs/PROGRESS.md` | Current state, phase by phase — read this first |
| `SPEC.md` | The full design |
| `docs/feature_book.md` | What every computed feature means |
| `docs/quality_report.md` | The latest data quality findings |
| `docs/adr/` | Why each design decision was made |
| `reviews/PHASE_0..4_REVIEW.md` | The review gate for each phase |
