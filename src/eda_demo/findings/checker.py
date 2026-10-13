"""The deterministic claim checker: a model's sentence is verified against tool results.

Design decision embodied here: **the check is arithmetic and textual, never another model.** A
second LLM asked "is this number right?" is just a second chance to be wrong in the same way. Three
mechanical rules cover the failure modes that matter:

1. every cited `result_id` must exist **in this run's store** — which is why the store is per run;
2. every number written in the claim must appear in the cited results' stats or parameters, within
   a rounding tolerance, so an invented figure cannot survive;
3. advisory language ("buy", "profitable", "guaranteed") rejects the finding outright, and causal
   language ("causes", "drives") caps its status at `observed`, because those are the two ways an
   EDA demo most easily turns into trading advice.

Rejections are counted and shown in the report. Nothing is silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..tools.registry import ResultStore
from .schema import FindingDraft

# Advisory wording: a hard failure. The demo never says what to do with money.
ADVISORY = re.compile(
    r"\b(buy|sell|short it|go long|profitable|profit from|guaranteed|risk[- ]free|"
    r"will rise|will fall|will go up|will go down|sure thing|must own|should own|"
    r"recommend (buying|selling)|position size)\b",
    re.IGNORECASE,
)
# Causal wording: a downgrade. The data cannot identify a cause.
CAUSAL = re.compile(
    r"\b(causes?|caused|causing|drives?|drove|driven|leads? to|led to|because of the tweets|"
    r"makes? the price|due to the sentiment|proves?)\b",
    re.IGNORECASE,
)
# A number with optional sign, decimals, thousands separators and a percent sign.
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?\s?%?")

# A citation tag, as required by SPEC.md 4.11: every numeric sentence carries `[R:<result_id>]`.
# It has to be stripped before numbers are scanned, because a result id is made of digits and would
# otherwise be read as a claim the moment the handle is inside it.
CITATION = re.compile(r"\[R:[^\]]*\]")

# "95% CI" / "95% confidence" is a confidence *level*, not a statistic: the interval itself is in
# the cited stats, so the level is checked by construction rather than by a number match. The
# percent sign is optional here because the number pattern may already have consumed it.
CI_LEVEL = re.compile(r"^\s*%?\s*(ci\b|confidence\b)", re.IGNORECASE)

# Dates, horizons and window years are legitimate in prose without being stats.
_YEAR = re.compile(r"^(19|20)\d{2}$")


@dataclass
class Verdict:
    """The checker's decision for one draft."""

    verdict: str          # accepted | downgraded | rejected
    reason: str = ""
    action: str = ""
    status_cap: str | None = None
    limitations: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.limitations is None:
            self.limitations = []


def _allowed_numbers(results: Iterable[Any]) -> set[float]:
    """Collect every number a claim may cite: result stats plus the parameters that produced them.

    Parameters are included because "h=1" or "3 buckets" are legitimate, verifiable statements
    about the call — they come from the registry, not from the model's imagination.
    """
    allowed: set[float] = set()
    for result in results:
        allowed.update(result.cited_numbers())
        for value in result.params.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                allowed.add(float(value))
            elif isinstance(value, list):
                allowed.update(float(item) for item in value if isinstance(item, (int, float)))
        allowed.add(float(result.n_units))
        allowed.add(float(result.n_obs))
        allowed.add(float(result.n_dropped))
    return allowed


def _match_tolerance(decimals: int) -> float:
    """Return how far a claim written to `decimals` may sit from a statistic.

    The claim's own precision defines the tolerance: "0.60" may describe 0.603786, while "0.9871"
    may not, and neither may "0.01" describe a zero. This is half a unit in the last written place,
    which is exactly the rule "the number rounds to a value the tool produced".
    """
    return 0.5 * (10.0 ** -min(decimals, 8)) + 1e-12


def _matches(number: float, allowed: set[float], decimals: int, percent: bool) -> bool:
    """True when `number` rounds to an allowed value at the precision the claim used.

    A percent-signed claim is also compared against the fraction form ("28.5%" against 0.285), which
    is the one scale mismatch ordinary prose produces. Nothing else is rescaled: a checker that
    accepted any near-miss would pass the invented numbers it exists to catch.
    """
    tolerance = _match_tolerance(decimals)
    candidates = [number, number / 100.0] if percent else [number]
    for candidate in candidates:
        for value in allowed:
            if abs(candidate - value) <= tolerance:
                return True
    return False


def _decimals_of(token: str) -> int:
    """Return the number of decimal places the claim was written with (capped at 8)."""
    cleaned = token.rstrip("%").replace(",", "").strip()
    if "." not in cleaned:
        return 0
    return min(len(cleaned.split(".", 1)[1]), 8)


def strip_citations(text: str, known_ids: Iterable[str] = ()) -> str:
    """Remove `[R:<id>]` tags and bare result ids so their digits are not read as claims.

    Also accepts the ids themselves, because a model sometimes writes the id without the tag and a
    bare 12-character hex string is never a statistic.
    """
    stripped = CITATION.sub(" ", text)
    for result_id in known_ids:
        if result_id:
            stripped = stripped.replace(result_id, " ")
    return stripped


def check_draft(
    draft: FindingDraft,
    store: ResultStore,
    known_numbers: set[float] | None = None,
) -> tuple[Verdict, list[Any]]:
    """Check one draft against this run's store. Returns (verdict, cited results).

    The cited results are returned so the caller does not have to look them up a second time (and
    cannot accidentally look them up in a different store).
    """
    limitations: list[str] = []

    if not draft.evidence:
        return Verdict("rejected", "no evidence: a finding must cite at least one result_id"), []

    cited = []
    for item in draft.evidence:
        result = store.get(item.result_id)
        if result is None:
            return (
                Verdict(
                    "rejected",
                    f"cited result_id '{item.result_id}' does not belong to this run "
                    f"(known ids: {store.order[:6]})",
                ),
                [],
            )
        cited.append(result)

    if ADVISORY.search(draft.claim_text):
        return (
            Verdict(
                "rejected",
                "advisory language found ("
                + ", ".join(sorted({match.group(0).lower() for match in ADVISORY.finditer(draft.claim_text)}))
                + "): this demo never gives trading advice",
            ),
            cited,
        )

    allowed = _allowed_numbers(cited)
    if known_numbers:
        allowed |= known_numbers

    unmatched: list[str] = []
    known_ids = [item.result_id for item in cited]
    claim_text = strip_citations(draft.claim_text, known_ids)
    for match in NUMBER.finditer(claim_text):
        token = match.group(0).strip()
        if CI_LEVEL.match(claim_text[match.end():]):
            continue
        cleaned = token.rstrip("%").replace(",", "").strip()
        if not cleaned or not re.match(r"^[-+]?\d", cleaned):
            continue
        try:
            number = float(cleaned)
        except ValueError:
            continue
        if _YEAR.match(cleaned):
            continue  # a window year, verified by the manifest rather than by a tool stat
        if not _matches(number, allowed, _decimals_of(token), token.rstrip().endswith("%")):
            unmatched.append(token)
    if unmatched:
        return (
            Verdict(
                "rejected",
                "claim cites number(s) not present in the cited stats: " + ", ".join(unmatched[:6]),
                action="drop the sentence or cite a result that contains the number",
            ),
            cited,
        )

    causal = sorted({match.group(0).lower() for match in CAUSAL.finditer(draft.claim_text)})
    if causal:
        limitations.append(
            "claim wording implies causation ("
            + ", ".join(causal)
            + "); this dataset cannot identify a cause, so the status is capped at observed"
        )
        return (
            Verdict(
                "downgraded",
                "causal language: " + ", ".join(causal),
                action="status capped at observed; limitation attached",
                status_cap="observed",
                limitations=limitations,
            ),
            cited,
        )

    return Verdict("accepted", "every cited id belongs to this run and every number matches a cited stat"), cited


def check_numbers_in_text(text: str, allowed: set[float], known_ids: Iterable[str] = ()) -> list[str]:
    """Return the numbers in `text` that match nothing in `allowed` (used for the narrative).

    The presenter's prose is checked the same way as a finding's claim, because the executive
    summary is the part most likely to be read on its own. `known_ids` are stripped first: a
    citation tag is a reference, not a quantity.
    """
    unmatched: list[str] = []
    text = strip_citations(text, known_ids)
    for match in NUMBER.finditer(text):
        token = match.group(0).strip()
        if CI_LEVEL.match(text[match.end():]):
            continue
        cleaned = token.rstrip("%").replace(",", "").strip()
        if not re.match(r"^[-+]?\d", cleaned) or _YEAR.match(cleaned):
            continue
        try:
            number = float(cleaned)
        except ValueError:
            continue
        if not _matches(number, allowed, _decimals_of(token), token.rstrip().endswith("%")):
            unmatched.append(token)
    return unmatched
