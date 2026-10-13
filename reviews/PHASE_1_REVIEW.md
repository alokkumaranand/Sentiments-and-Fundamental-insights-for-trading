# Phase 1 review — LLM wiring + Feature Induction Agent

**Gate: awaiting your review.** Phase 2 has been built on top of it; nothing here is provisional in
the sense of being untested, but the two decisions in §5 are yours whenever you want to revisit them.

Phase 1 had to end with "a live endpoint returns a proposal, the gate accepts or rejects with recorded
reasons, the frozen lexicon is written, and with no key the whole path falls back to `default_v1.json`".
All four are demonstrated below on the real corpus.

---

## 1. What was built

| File | Purpose |
|---|---|
| `src/eda_demo/agents/llm.py` | the only LLM touchpoint: OpenAI-compatible `/chat/completions` on `urllib`, response cache, native tool calling with an automatic `json_prompt` fallback, `ping()` for `doctor` |
| `src/eda_demo/agents/sampler.py` | the capped, stratified, deterministic sample and the session-disjoint validation slice |
| `src/eda_demo/agents/induction.py` | the prompt, the `LexiconProposal` run, the deterministic validation gate, the frozen lexicon, `rejected_proposals.json` |
| `src/eda_demo/features/lexicon.py` | the ledger contract and the drift-coverage measure |
| `src/eda_demo/features/definitions.py` | every feature's definition and caveat, as data (used from Phase 2 on) |
| `config/lexicons/default_v1.json` | the shipped no-key lexicon: 162 hand-written terms across all six families |
| `run.py doctor` | now sends a 1-token reachability ping and reports it as a warning, never a failure |
| `run.py induce` | the smoke test you can re-run; `--no-live` demonstrates the fallback path |
| `tests/test_phase1.py` | 21 tests |

Designed in from the start: the key exists only in the request header; the manifest records the
base-URL **host** and model, never the key.

## 2. How to verify it yourself

```powershell
.\.venv\Scripts\python.exe run.py doctor          # now includes the 1-token LLM ping
.\.venv\Scripts\python.exe run.py --dataset full induce        # live call + gate
.\.venv\Scripts\python.exe run.py --dataset full induce --no-live   # fallback path, no call
.\.venv\Scripts\python.exe run.py --dataset dev playbook --no-menu  # a run that never needs a key
.\.venv\Scripts\python.exe -m pytest             # 118 tests total
```

## 3. Evidence — the live smoke test on the full corpus

| | |
|---|---|
| Endpoint | `api.experientiallabs.ai`, model `gpt-5.6-luna`, native tool calling detected, temperature 0 |
| Sample sent | **1,375 tweets across all 25 tickers** (promo 865 · cashtag-dense 228 · ordinary 212 · reply 70) |
| Sample prompt | 84,117 prompt tokens, one call, $0.023, cached by prompt hash |
| Validation slice | **40,904 tweets across 126 sessions — session-disjoint from the sample** |
| Proposed / accepted / rejected | **47 / 2 / 45** |
| Accepted | `buy the dip` (+0.65), `media report` (attribution marker) |
| Frozen lexicon | `artifacts/lexicons/8ad5ddd16848.json`, 164 terms, active pointer written |
| Rejection reasons | 43 × `support < 10`, 1 × `bootstrap CI could not be estimated`, 1 × `CI includes zero`; every one is named in `rejected_proposals.json` |

That ratio is the point of the phase: a model's list is a *hypothesis list*, and 45 of its 47
hypotheses did not survive contact with the held-out slice. The two that did are auditable — the
frozen file carries each term's support, sign agreement and CI.

## 4. Two problems the smoke test exposed, and what changed

1. **The first live call returned truncated JSON.** `llm.max_tokens` was 2000 and a 60-term proposal
   does not fit. Two changes: the induction call now has its own budget
   (`agents.induction.max_tokens: 8000`), and the client *salvages* a truncated answer by keeping the
   longest prefix that ends at a **completed** value, recording `repaired: true`. A half-written number
   is never completed into a value — dropping a partial element is honest, inventing `2` from a cut-off
   `[1,2000` is not.
2. **A CI that cannot be estimated was passing the gate.** With one occurrence the bootstrap returned
   the point estimate as its own interval, which reads as certainty. Now fewer than two session blocks
   yields `[nan, nan]` and the gate **rejects**: a gate that passes when it cannot measure is not a gate.

## 5. What I need from you (the Phase 1 gate)

| # | Item | My recommendation |
|---|---|---|
| 1 | **The gate's notion of truth.** It measures agreement with the polarity of the *surrounding text* under the shipped lexicon, because this corpus has no sentiment labels. It therefore validates "agrees with a general sentiment prior", not "correct on finance slang". See ADR-0006. | Accept, with the limitation stated in every review and in the frozen file's own notes. The alternative is hand-labelling tweets, which this demo is explicitly not doing. |
| 2 | **The sample size.** 1,375 tweets is an 84k-token prompt and ~$0.02 per induction. It covers all 25 tickers at a 60-tweet cap in theory, but the strata caps fill it with promo text (865 of 1,375). | Keep it, but if you want a cheaper/cleaner call, lower `agents.induction.max_total` to ~600 and reduce `strata_caps.promo` to 0.10. One config edit, no code change. |
| 3 | **Is the induced lexicon active?** It is now: the pointer `artifacts/lexicons/active.json` selects `8ad5ddd16848`, and `run.py features --lexicon default` reverts for one run. | Keep it active — that is what makes `lexicon_id` meaningful — but note that activating a vocabulary changes every sentiment number, by design. |
| 4 | **`gpt-5.6-luna` honoured temperature 0 per call, but two identical explorer runs still chose different tools.** The seed vocabulary call is cached, so a repeat costs nothing, but agent *choices* are not reproducible. | Accept and state it: only the deterministic layer is reproducible. The cache is what makes the common case cheap; the manifest records the mode and the prompt hash either way. |

## 6. What Phase 2 did with this

`sentiment.method: vader_plus_lexicon` blends VADER with the active lexicon
(`lexicon_weight: 0.5`); negation flips and damps a term within 3 tokens; sarcasm is reported as a
caveat share rather than modelled. The lexicon id is part of every `result_id` through
`data_version`, so a vocabulary change is visible in every downstream number.
