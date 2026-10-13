# Phase 4 review — findings, claim checker, agents, manifest

**Gate: awaiting your review.** Phase 4 is the last required phase. It had to end with "agent findings
pass the claim checker; failures degrade gracefully to the templated report; the manifest reproduces
the run". All three are evidenced below, on the full corpus, with the live endpoint.

---

## 1. What was built

| File | Purpose |
|---|---|
| `src/eda_demo/findings/schema.py` | `FindingDraft` (what an agent may return) and `Finding` (what is presented) — plus `findings.json`'s shape |
| `src/eda_demo/findings/checker.py` | the deterministic claim checker: cited ids, cited numbers, advisory and causal wording |
| `src/eda_demo/findings/status_rules.py` | status by rule, with the tier-C and tier-B structural caps |
| `src/eda_demo/agents/null_agent.py` | the no-key path: findings selected by a written-down rule, templated prose |
| `src/eda_demo/agents/explorer.py` | the tool-calling loop, with the discovery-only guardrail enforced in code |
| `src/eda_demo/agents/presenter.py` | the narrative, re-checked against the numbers by the same arithmetic rule |
| `src/eda_demo/pipeline.py` | the coordinator: stages, holdout + BH, claim checking, status, manifest |
| `src/eda_demo/append.py` | the new-data path: append-only, deterministic re-scoring, drift check (SPEC.md 7.3) |
| `run.py run` · `run.py append` | the full pipeline and the new-data entry point |
| `tests/test_phase4.py` | 27 tests |

## 2. How to verify it yourself

```powershell
.\.venv\Scripts\python.exe run.py --dataset dev  run --no-menu       # fast, agents on
.\.venv\Scripts\python.exe run.py --dataset full run                 # the run described below
.\.venv\Scripts\python.exe run.py --dataset full run --no-agents     # templated path, no model call
.\.venv\Scripts\python.exe run.py --dataset full append --file arriving.csv
.\.venv\Scripts\python.exe -m pytest tests/test_phase4.py
```

## 3. Evidence — the end-to-end agent run on the full corpus

| | |
|---|---|
| Lexicon | `8ad5ddd16848` (induced in Phase 1, frozen, reused — induction was **skipped**, as designed) |
| Tool calls | **43 results**, 12 of them chosen by the explorer within its budget |
| LLM calls | 4 (explorer ×3, presenter ×1), native tool calling for the loop, JSON mode for the drafts |
| Findings | **6**: 2 `supported`, 2 `inconclusive`, 2 `observed` |
| Multiplicity | `adjust_pvalues` over **11 tests**, BH, alpha 0.05 → **2 significant** |
| Holdout | 8 coordinator-triggered `holdout_check` re-runs, each recorded with the child `result_id` |
| Claim checker | **0 rejections** — every paragraph and every citation passed |
| Narrative | `presenter_agent`; each numeric sentence carries its `[R:<result_id>]` |
| Runtime | 60.5 s with the panel cached (panel rebuild adds ~40 s) |

The two `supported` findings are the same honest result seen twice: **specific attention and *same-day*
absolute return**, pooled excluding TSLA — rho 0.1108 (95% CI 0.0255–0.1946, q 0.0068) and rho 0.1115
(q 0.0045), both with the same sign on the holdout. The sentiment → forward-return tests are
**inconclusive** at every horizon (q 0.76 and 0.96), which is the honest headline: on this corpus the
attention family carries a signal and the sentiment family does not.

Every finding card in `findings.json` records its evidence ids, its effect, its limitations, the
`claim checker` verdict, and its origin (`explorer_agent`).

## 4. What the claim checker actually caught

The checker's rejections during development are worth reporting, because they are what the phase is for:

1. **The model's own citation tags were read as invented numbers.** `[R:8bf00a274337]` contains digits,
   so the number scan flagged them. Fixed by stripping `[R:…]` and known ids before scanning — and that
   bug would have made every model-written narrative get thrown away, silently degrading the demo to
   templates while *looking* like it was working.
2. **The tolerance was too loose.** "Within 1% or 0.01" accepted a fabricated `0.9871` because the
   percent-scaled candidate (`0.009871`) sat within 0.01 of a zero in the stats. The rule is now "the
   claim must round to a value the tool produced, at the precision the claim was written with":
   `0.60` may describe 0.603786; `0.9871` may not.
3. **`holdout_check` silently re-ran discovery.** The tool reads `params.period`, so passing
   `period="holdout"` to the registry alone changed the *label* and not the computation. Every
   replication check would have been a false pass. The registry now makes its `period` argument
   authoritative and overwrites the parameter; a test guards it.
4. **A cached panel differed from a fresh one** by one ULP in some float columns, which moved a rank
   correlation in the fourth decimal. Fixed with 17-digit output plus pandas' round-trip parser
   (ADR-0007); a test asserts the cache round-trips exactly.

## 5. Status rules, as implemented

| Status | Rule applied in `findings/status_rules.py` |
|---|---|
| `observed` | descriptive, or a structural cap (tier C; tier B without own-text restriction) |
| `supported` | BH q < alpha **and** CI excludes zero **and** same sign on the holdout |
| `not_replicated` | q < alpha and CI excludes zero on discovery, sign flipped on the holdout |
| `inconclusive` | tested, but none of the above (CI includes zero, or q ≥ alpha, or no q) |

A `supported` finding's limitation text says explicitly that replicating on a 30% holdout inside one
regime is weak evidence. Nothing is called predictive, and advisory wording rejects a finding outright.

## 6. The no-key path is the same pipeline

`run --no-agents` (or no key configured) selects findings by rule — largest absolute bucket spread,
largest absolute correlation, the coverage baseline, the biggest spike count, text hygiene — and writes
the same artifacts. A test asserts that every templated draft passes the same claim checker, which is
the strongest available evidence that the checker is arithmetic rather than deference.

## 7. New data (SPEC.md 7.3)

`run.py append --file <csv>`:

1. **append-only** — existing rows are never rewritten; the new file is read, cleaned and scored into
   a new run directory;
2. **deterministic re-scoring** with the active frozen lexicon, **0 LLM calls** (asserted in the report);
3. **drift check** — `uncovered_token_share` against `quality.max_uncovered_token_share`, reported with
   examples, and a re-induction recommendation when it is exceeded on enough new text;
4. **schema drift is blocking** — a renamed or missing column stops the run with the diff, because
   guessing which column is the text would corrupt every number.

The path is exercised on a fixture in the tests; there is no second price file to extend the panel past
the existing window, so appended rows inside the window are placed on sessions and rows past the last
session are counted and flagged rather than guessed at.

## 8. Decisions at this gate

| # | Item | Recommendation |
|---|---|---|
| 1 | **Agent choices are not reproducible.** Two runs with identical prompts (temperature 0) chose different tools and produced 4 vs 6 findings. The deterministic layer is identical; the finding set is not. | Accept and state it: the manifest records the mode and prompt hash, the numbers are all tool-sourced, and `--no-agents` gives the reproducible report. |
| 2 | **Critic agent** (spec decision 8) | Left out, as agreed: `agents.critic.enabled` does not exist yet. The claim checker is not a critic; it is arithmetic. |
| 3 | **Findings count** | `agents.presenter.max_findings: 8`. The explorer asked for 6 and got them. |
| 4 | **`report.html`** | Convenience export; `report.md` and `findings.json` are the durable artifacts. |

## 9. Where the demo stands as a whole

All five phases are complete, all 120 tests pass, and the acceptance criteria in SPEC.md §10 hold:
read-only ingestion, canonical tables, a complete no-key report, three or more §8 analyses with
`n_units` and `n_dropped`, forward returns with a chronological holdout, `result_id` on every figure and
finding, schema-validated agent output with every cited number checked, badges that separate observed
data from estimates from interpretation, a hashed induced lexicon, an append path with no LLM call, and
a README that states setup, assumptions and limitations.
