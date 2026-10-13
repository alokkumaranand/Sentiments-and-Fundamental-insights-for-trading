# Phase 3 review — tool library, baseline playbook, terminal presentation

**Gate: awaiting your review.** Phase 3 had to end with "the playbook runs end to end with no LLM,
`ToolResult`s are persisted, charts render from stored specs, and every figure caption carries its
`result_id`". All four are evidenced below.

---

## 1. What was built

| File | Purpose |
|---|---|
| `src/eda_demo/tools/schema.py` | `ToolResult` and `ChartSpec`, the two contracts (SPEC.md 4.8, 4.9) |
| `src/eda_demo/tools/registry.py` | schema-validated parameters, `result_id`, the result store, the standard caveats and the caption |
| `src/eda_demo/tools/descriptive.py` | `coverage_by_ticker`, `distribution`, `text_stats`, `timeseries`, `sample_tweets` |
| `src/eda_demo/tools/relational.py` | `correlate`, `bucket_forward_returns`, `corr_matrix`, `spike_days`, `attention_vs_price`, `sentiment_vs_price` |
| `src/eda_demo/tools/validation.py` | `holdout_check`, `adjust_pvalues` |
| `src/eda_demo/tools/playbook.py` | the fixed 14-call baseline, as data |
| `src/eda_demo/stats/inference.py` | session-block bootstrap, Spearman, Benjamini-Hochberg |
| `src/eda_demo/report/render_ascii.py` · `render_png.py` | two renderers over the same `ChartSpec` |
| `src/eda_demo/report/chart_menu.py` | the numbered menu, with `3,5` / `all` / `ascii 2` / `md` / `log` / `q` |
| `src/eda_demo/report/build_report.py` | `report.md` (primary), `report.html` (self-contained export), the terminal overview |
| `run.py playbook` · `run.py charts` | the no-LLM entry point and the menu for a finished run |
| `tests/test_phase3.py` | 26 tests |

**The catalog is the whole surface.** 14 tools; nothing else is callable by an agent. `sample_tweets`
is capped at 10 rows and is the only tool that returns raw corpus text.

## 2. How to verify it yourself

```powershell
.\.venv\Scripts\python.exe run.py --dataset dev  playbook          # interactive menu at the end
.\.venv\Scripts\python.exe run.py --dataset full playbook
.\.venv\Scripts\python.exe run.py --dataset full charts --run-id <run_id>
.\.venv\Scripts\python.exe -m pytest tests/test_phase3.py
```

In the menu: a number renders that chart to `artifacts/runs/<run_id>/figures/<result_id>.png` and opens
it; `ascii 2` prints it in the terminal; `all` writes every PNG without opening windows.

## 3. Evidence

- **Baseline runs with no endpoint.** `playbook` makes zero LLM calls by construction (the mode never
  constructs an agent call), produces **14 tool calls with 0 skipped** on both `dev` and `full`, and
  writes `results/results.json`, `report.md`, `report.html` and the terminal overview.
- **Every caption is traceable.** The registry, not the tool, appends `; result <result_id>;
  data_version=…` to each caption; a test asserts period, `n_units` and the id are present on a
  produced chart.
- **Both renderers accept every kind.** A test renders all ten chart-bearing tools through the ASCII
  renderer and to PNG files (all seven `ChartSpec` kinds are exercised), so a spec that leaked a
  matplotlib concept would fail there rather than in front of you.
- **Pooling is never alone.** The playbook always runs the pooled pair *and* `pooled_ex_TSLA`; a test
  asserts the excluded variant is present in the baseline's scopes.
- **A call that cannot run is recorded, not fatal.** On the 26-session `dev` slice some tools have too
  few units; the manifest lists them under `playbook_failures` with the reason, and the run completes.

## 4. Statistical defaults (all in `config/experiments.yaml`)

| Setting | Value | Why |
|---|---|---|
| `bootstrap.block_sessions` | 5 | one trading week; h>1 forward returns overlap, so an IID bootstrap is too narrow |
| `bootstrap.resamples` | 2000 | fixed seed, so a CI is reproducible |
| `multiple_testing.alpha` · method | 0.05 · Benjamini-Hochberg | screening 25 tickers × 3 horizons is 75 tests; ~4 would look significant by chance |
| `attention.baseline_sessions` | 20 | one trading month, shifted so day *d* sees only earlier days |
| `tools.table_row_cap` | 50 | an agent cannot paste a dataset into a report |

## 5. What is deliberately not here

- **No insight registry, no web UI, no interactive dashboard** — the menu is the whole UI, and Markdown
  is the durable artifact (ADR-0004).
- **No figure objects in tools.** A tool returns a `ChartSpec`; rendering is a separate, swappable step.
- **No automatic chart generation.** Nothing renders unless you ask for it by number, so a run never
  opens twelve windows.

## 6. Decisions at this gate

| # | Item | Recommendation |
|---|---|---|
| 1 | Chart backend | PNG via the OS viewer by default, `ascii <n>` always available. `charts.default_backend` switches it. |
| 2 | Is `report.html` primary? | No — a convenience export that embeds the PNGs that exist. `report.md` is primary (spec decision 9). |
| 3 | `all` opens no windows | Rendering everything at once should not spawn a dozen viewers; a single number opens. |
| 4 | TSLA-excluded pooling | Mandatory wherever "pooled" appears, and the caveat says so on the result. |

## 7. What Phase 4 adds on top

Findings, the deterministic claim checker, the status rules and the two agents that read this tool
library. Nothing in this phase changes: the agents can only call these tools, and the coordinator — not
the explorer — triggers `holdout_check` and `adjust_pvalues`.
