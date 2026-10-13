# 0001. Dataset slices and where the paths live

**Status:** Accepted · **Date:** 2026-10-12

## Context

The raw corpus is 80,793 tweets and 6,300 price rows. The human split both files 1:9
(chronological, earliest 10% = `validation`) to avoid consuming the whole dataset while the
system is being built, in case the work stalls. Two constraints came with that:

- build and test against the small slice, not the full corpus;
- it must be obvious where to change the path to use a different dataset.

Separately, the small slice turned out to have only 26 trading sessions and a ragged final
session (5 of 25 tickers), so it cannot support any statistical conclusion.

## Decision

Select the dataset in **one** place: `config/settings.yaml` → `datasets.active`, with every
source defined under `datasets.sources`. Override for a single run with `--dataset <name>`.
The section is commented as the place to edit, and adding a new dataset is a config-only change.
`Documents`/paths are resolved against the repository root by `src/eda_demo/config.py`, the only
module that reads configuration.

`dev` is the default (small slice, fast, correctness only). `full` is required for anything
reported as a finding.

## Consequences

- Build and test loops stay fast, and the untouched slices remain available.
- **No conclusion may be drawn from a `dev` run.** The profiler states this in a warning, and
  `PROGRESS.md` repeats it. Anyone reporting a `dev` number has ignored a written warning.
- A future session must not "helpfully" default to `full` to make results look better; the
  slice in use is recorded in every run manifest and every `data_version`.
- Tier counts differ between slices (dev A=2, full A=3) because own-text share is
  window-dependent. Compare tiers only within the same slice.
