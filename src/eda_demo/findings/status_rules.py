"""Status is assigned by rules, never by the LLM (SPEC.md 4.11).

The four statuses mean exactly this and nothing more:

  observed        a descriptive result with no hypothesis test behind it
  supported       tested: BH-adjusted q below alpha, CI excludes zero, and the sign held on the holdout
  not_replicated  it passed on discovery and failed or flipped on the holdout
  inconclusive    tested, but none of the above

Two overrides are structural rather than statistical, and both exist because the dataset says so:

- a **tier C** ticker's text belongs to other tickers, so a per-ticker finding about it can never
  exceed `observed`;
- a **tier B** per-ticker finding is capped at `observed` unless its text was restricted to own
  text, because otherwise the number is partly about the shared chatter pool.

`supported` never implies a trading edge. The wording is deliberately the weakest that is true.
"""

from __future__ import annotations

from typing import Any


def assign_status(
    *,
    has_test: bool,
    q_value: float | None,
    ci_95: list[float | None] | None,
    same_sign_on_holdout: bool | None,
    tier: str | None,
    own_text_only: bool,
    alpha: float,
) -> tuple[str, list[str]]:
    """Return (status, extra_limitations) for one finding.

    Pure function of the numbers the coordinator computed, so the rule can be unit-tested without
    a dataset — which matters, because status is the one word a reader is most likely to over-read.
    """
    limitations: list[str] = []

    def cap(status: str, reason: str) -> tuple[str, list[str]]:
        """Force the status down to `observed`, recording why so nothing is silently weakened.

        The cap is absolute rather than relative: these two overrides are about the data not being
        able to answer the question at all, and `observed` is the only status that claims nothing.
        """
        limitations.append(reason)
        return "observed", limitations

    if tier == "C":
        return cap(
            "observed",
            "coverage tier C: every row of this ticker is cross-posted text, so the measure "
            "describes the generic chatter pool rather than the company",
        )
    if tier == "B" and not own_text_only:
        return cap(
            "observed",
            "coverage tier B without own-text restriction: at most half of this ticker's text is "
            "its own, so a per-ticker inference is not ticker-specific",
        )

    if not has_test:
        limitations.append("descriptive result: no hypothesis test was run, so no status above observed applies")
        return "observed", limitations

    if q_value is None or ci_95 is None or ci_95[0] is None or ci_95[1] is None:
        limitations.append("tested, but the CI or p-value could not be estimated on this sample")
        return "inconclusive", limitations

    excludes_zero = (ci_95[0] > 0) or (ci_95[1] < 0)
    if q_value < alpha and excludes_zero:
        if same_sign_on_holdout is True:
            limitations.append(
                "replicated on the chronological holdout, which is 30% of sessions in the same "
                "single regime: replication here is weak evidence, not confirmation"
            )
            return "supported", limitations
        if same_sign_on_holdout is False:
            limitations.append("the sign flipped on the holdout period")
            return "not_replicated", limitations
        limitations.append("the holdout period could not be checked for this result")
        return "inconclusive", limitations

    if not excludes_zero:
        limitations.append("the 95% CI includes zero, so the direction is not resolved")
    if q_value >= alpha:
        limitations.append(
            f"BH-adjusted q = {q_value:.3f} is not below alpha = {alpha}, so this survives neither "
            "the correction for the number of tests nor the significance bar"
        )
    return "inconclusive", limitations


def status_badge(status: str) -> str:
    """Return the presentation badge for a status (SPEC.md 4.12)."""
    if status == "observed":
        return "OBSERVED DATA"
    if status == "supported":
        return "STATISTICAL ESTIMATE"
    return "STATISTICAL ESTIMATE"


def summarise_status_counts(findings: list[Any]) -> dict[str, int]:
    """Count findings per status, for the manifest and the terminal overview."""
    counts = {status: 0 for status in ("observed", "supported", "not_replicated", "inconclusive")}
    for finding in findings:
        status = finding["status"] if isinstance(finding, dict) else finding.status
        counts[status] = counts.get(status, 0) + 1
    return counts
