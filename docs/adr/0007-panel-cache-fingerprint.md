# 0007. The cached panel is keyed by a fingerprint that is checked on every load

**Status:** Accepted · **Date:** 2026-10-12

## Context

The feature layer is the slow stage: it cleans and scores every tweet, then builds the ticker-session
panel. Phases 3 and 4 run tools over that panel repeatedly, so it is cached under `artifacts/cache/`.

The first implementation wrote a `panel_latest_<dataset>.txt` pointer holding the panel's
`data_version`, and loaded whatever that pointer named. Two problems appeared immediately in testing:

- editing a **column** the cache holds (adding `cashtag_count`) left the pointer valid, so the next
  run loaded a table missing the column a tool needed — a `KeyError` from a stage that should have
  rebuilt;
- a pointer is only as fresh as the last write, so any change to config or code that a tool depends
  on could be served a panel computed under the old settings.

This is the class of bug that produces a plausible wrong number rather than a crash, which is the
worst kind for a demo about traceability.

## Decision

`save_panel` writes a JSON pointer holding both the `data_version` and a **fingerprint**:
sha256 of the two input *container* hashes, both config files, the lexicon id, the code version and a
`CACHE_SCHEMA` integer that is bumped whenever the cached column set changes.

`get_panel` recomputes the fingerprint before using the cache and treats a mismatch — or a pointer in
an older format — as "no cache": it rebuilds. The lexicon is resolved *before* the cache check, because
a panel scored with the shipped lexicon is a different table from one scored with an induced
vocabulary, and the fingerprint has to include `lexicon_id` for that distinction to hold.

The fingerprint hashes the container files, not the decompressed zip member: a change anywhere in the
file changes the container hash, and hashing 6 MB takes tens of milliseconds instead of a full CSV
parse.

## Consequences

- A stale panel cannot be served. The cost is a few tens of milliseconds of hashing per run.
- `--no-cache` still exists for measurement, and the cache remains an optimisation: a missing or
  unreadable one rebuilds rather than failing.
- The cache is written with `float_format="%.17g"`. Default CSV formatting lost the last bits of some
  float64 columns, which was visible as a rank correlation differing in the fourth decimal between a
  cached and a freshly built panel. 17 significant digits round-trips a float64 exactly, so the cache
  is now bit-identical to the computation it caches — which is the only version of "identical stats"
  that is worth claiming.
- Anything that changes what the cache *means* must bump `CACHE_SCHEMA` or be part of the fingerprint.
  If a future stage adds a setting that affects the panel, add it to the fingerprint payload in
  `panel_data_version`/`cache_fingerprint` **in the same commit**.
- Result ids already covered data and config; this record extends the same principle to the cache, so
  "the cache is provably the same computation" is a claim with a mechanism behind it.
