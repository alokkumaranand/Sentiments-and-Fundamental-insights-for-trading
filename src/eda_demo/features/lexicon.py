"""The lexicon contract: schemas, loading, hashing and drift coverage.

Design decision embodied here: the LLM never scores anything. Its only low-level output is a
`LexiconProposal` — a list of candidate terms with a claimed polarity — which is validated,
frozen to JSON and hashed. Deterministic code then applies that frozen file to the whole
corpus. Keeping the schema in one module (with `extra="forbid"`) is what makes the frozen
artifact reviewable and what makes an unexpected extra key from a model a loud failure rather
than a silently ignored field.

`default_v1.json` is shipped and hand-written, so nothing downstream ever depends on the
induction stage having run or on an endpoint being reachable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..config import REPO_ROOT, Settings

# The families a term may belong to. `polarity` is general signed sentiment, `bullish` and
# `bearish` are the finance directional list, and the last three are scored as shares
# (hedging, reported speech, sarcasm), never as polarity.
Family = Literal["polarity", "bullish", "bearish", "uncertainty", "attribution", "sarcasm"]

# Tokens ignored by the drift-coverage check: they carry no sentiment and would otherwise
# inflate "uncovered" for every ordinary English sentence.
STOPWORDS = frozenset(
    """a an and are as at be been but by for from had has have he her his i if in into is it its
    me my no not of on or our out she so than that the their them then there these they this to
    too up us was we were what when which who will with you your""".split()
)


class Term(BaseModel):
    """One lexicon entry, plus the validation-gate evidence when it was induced.

    `evidence_tweet_ids` is what the inductor cited; `support`/`sign_agreement`/`ci_95` are
    filled in by the deterministic gate, so a reviewer can see why a term was accepted.
    """

    model_config = ConfigDict(extra="forbid")

    term: str = Field(min_length=1)
    family: Family
    polarity: float = Field(ge=-1.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    evidence_tweet_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    support: int | None = None
    sign_agreement: float | None = None
    ci_95: list[float] | None = None


class RelevanceRule(BaseModel):
    """A proposed rule for deciding whether text is about a ticker at all.

    Carried through induction for review; v0 applies none of them, because the measured
    own-text share already answers the question with data rather than a rule.
    """

    model_config = ConfigDict(extra="forbid")

    pattern: str
    action: str
    rationale: str = ""


class LexiconProposal(BaseModel):
    """Exactly what the inductor is allowed to return (SPEC.md 4.3), nothing more."""

    model_config = ConfigDict(extra="forbid")

    terms: list[Term] = Field(default_factory=list)
    relevance_rules: list[RelevanceRule] = Field(default_factory=list)
    negation_patterns: list[str] = Field(default_factory=list)
    sarcasm_markers: list[str] = Field(default_factory=list)
    abstain_reasons: list[str] = Field(default_factory=list)


class FrozenLexicon(BaseModel):
    """A lexicon that may be used to score the corpus, with its provenance.

    `source` is 'default' for the shipped file and 'induced' for one that passed the gate.
    Every result records `lexicon_id`, so a change in vocabulary is visible downstream.
    """

    model_config = ConfigDict(extra="forbid")

    lexicon_id: str
    source: Literal["default", "induced"]
    version: str = "1"
    created: str = ""
    sentiment_method: str = "vader_plus_lexicon"
    model: str = ""
    notes: list[str] = Field(default_factory=list)
    negation_patterns: list[str] = Field(default_factory=list)
    relevance_rules: list[RelevanceRule] = Field(default_factory=list)
    abstain_reasons: list[str] = Field(default_factory=list)
    terms: list[Term] = Field(default_factory=list)

    # --- accessors: the scorer asks for a family, never for a raw list -------

    def by_family(self, family: Family) -> list[Term]:
        """Return the terms of one family (empty list when there are none)."""
        return [t for t in self.terms if t.family == family]

    def polarity_terms(self) -> list[Term]:
        """Terms that carry signed sentiment: general polarity plus finance directional."""
        return [t for t in self.terms if t.family in ("polarity", "bullish", "bearish")]

    def markers(self, family: Family) -> set[str]:
        """Return the lowercased set of terms for a share-scored family."""
        return {t.term.lower() for t in self.by_family(family)}

    def all_terms_lower(self) -> set[str]:
        """Every term lowercased, used by the drift-coverage check."""
        return {t.term.lower() for t in self.terms}


def default_lexicon_path(settings: Settings) -> Path:
    """Return the shipped no-key lexicon path from `settings.yaml -> sentiment.lexicon`."""
    configured = (settings.raw.get("sentiment", {}) or {}).get("lexicon", "config/lexicons/default_v1.json")
    path = Path(str(configured).replace("<lexicon_id>", "default_v1"))
    return path if path.is_absolute() else REPO_ROOT / path


def load_lexicon(path: Path) -> FrozenLexicon:
    """Read a frozen lexicon from disk, raising a clear error on a schema problem.

    Fails loudly rather than repairing: a lexicon that does not validate must not score data.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return FrozenLexicon.model_validate(payload)
    except ValidationError as error:  # pragma: no cover - only on a hand-edited file
        raise SystemExit(f"invalid lexicon {path}:\n{error}") from error


def lexicon_id_for(terms: list[Term], induction_config: dict[str, Any], model_name: str, seed: int) -> str:
    """Hash the accepted terms plus the config, model and seed into a 12-char id.

    SPEC.md 4.3: `lexicon_id = sha256(canonical_json(accepted_terms) + induction_config +
    model_name + seed)[:12]`. Canonical means sorted keys and sorted terms, so the id depends
    on the vocabulary rather than on the order a model happened to emit it.
    """
    payload = json.dumps(
        {
            "terms": sorted((t.model_dump() for t in terms), key=lambda d: (d["term"], d["family"])),
            "induction": induction_config,
            "model": model_name,
            "seed": seed,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def write_lexicon(lexicon: FrozenLexicon, path: Path) -> Path:
    """Write a frozen lexicon as canonical JSON (sorted keys, indent 2) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lexicon.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def active_pointer_path(settings: Settings) -> Path:
    """Location of the tiny pointer that names the lexicon a run should use.

    Why a pointer exists: `artifacts/lexicons/` accumulates one file per induced vocabulary,
    and a run must pick one deterministically. The pointer records the id *and* the sentiment
    method it was built for; a method change invalidates it, so the shipped default is used
    instead of silently scoring with a vocabulary built for a different scorer.
    """
    return REPO_ROOT / "artifacts" / "lexicons" / "active.json"


def resolve_lexicon(
    settings: Settings, force_default: bool = False
) -> tuple[FrozenLexicon, dict[str, Any]]:
    """Return (lexicon, provenance) for this run: induced if pointed at, else the default.

    Provenance is a plain dict so it can be embedded in the run manifest and in
    `data_version`: {'source': 'induced'|'default', 'path': str, 'reason': str}.
    """
    method = str((settings.raw.get("sentiment", {}) or {}).get("method", "vader_plus_lexicon"))
    pointer = active_pointer_path(settings)
    if not force_default and pointer.exists():
        try:
            record = json.loads(pointer.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            record = {}
        candidate = Path(record.get("path", ""))
        if (
            record.get("sentiment_method") == method
            and candidate.exists()
            and candidate.is_relative_to(REPO_ROOT)
        ):
            return load_lexicon(candidate), {
                "source": "induced",
                "path": str(candidate),
                "lexicon_id": record.get("lexicon_id", ""),
                "reason": "active induced lexicon from artifacts/lexicons/active.json",
            }
    path = default_lexicon_path(settings)
    lex = load_lexicon(path)
    return lex, {"source": "default", "path": str(path), "lexicon_id": lex.lexicon_id,
                 "reason": "shipped default_v1 (no active induced lexicon for this method)"}


def set_active_lexicon(settings: Settings, lexicon: FrozenLexicon, path: Path) -> Path:
    """Point later runs at `path` as the lexicon to use. Called only by `induce`.

    Returns the pointer path. The pointer names the sentiment method as well as the file, so a
    later change of method invalidates it deliberately (see `resolve_lexicon`).
    """
    pointer = active_pointer_path(settings)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        json.dumps(
            {
                "lexicon_id": lexicon.lexicon_id,
                "path": str(path),
                "sentiment_method": lexicon.sentiment_method,
                "source": lexicon.source,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return pointer


def uncovered_token_share(texts: list[str], lexicon: FrozenLexicon) -> dict[str, Any]:
    """Return the share of content tokens the lexicon (plus VADER) does not cover.

    SPEC.md 7.3 step 3: this is the drift signal that decides whether new text still fits the
    frozen vocabulary. VADER's own lexicon counts as coverage because
    `vader_plus_lexicon` uses both, and ignoring it would make the shipped default look
    under-covering by construction.
    """
    covered = lexicon.all_terms_lower()
    try:  # optional dependency; absence only makes the measure more conservative
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        covered |= {word.lower() for word in SentimentIntensityAnalyzer().lexicon}
    except Exception:  # pragma: no cover - vaderSentiment is in requirements.txt
        pass

    seen: set[str] = set()
    uncovered: set[str] = set()
    for text in texts:
        for token in str(text).lower().split():
            token = token.strip(".,!?:;\"'()[]{}#@$%")
            if len(token) < 3 or token in STOPWORDS or not token.isalpha():
                continue
            seen.add(token)
            if token not in covered:
                uncovered.add(token)
    share = (len(uncovered) / len(seen)) if seen else 0.0
    return {
        "tokens_seen": len(seen),
        "tokens_uncovered": len(uncovered),
        "uncovered_token_share": round(share, 4),
        "examples": sorted(uncovered)[:20],
    }
