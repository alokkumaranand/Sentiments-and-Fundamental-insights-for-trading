"""Get the panel, rebuilding only when the cache cannot serve it.

Design decision embodied here: **stages communicate through files** (SPEC.md 3). The feature
layer is the slow part (80k tweets cleaned and scored), so it is cached under
`artifacts/cache/` keyed by the panel `data_version` — which covers the input hashes, both
config files, the lexicon id and the code version. A cached panel is therefore not "a stale
copy", it is provably the same computation.

The cache is an optimisation and never a dependency: a missing, partial or unreadable cache
falls back to rebuilding, and `--no-cache` forces a rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..features.lexicon import FrozenLexicon, resolve_lexicon
from ..features.panel import Panel, build_panel, cache_fingerprint, load_cached_panel, save_panel
from .align import build_canonical


@dataclass
class PanelResult:
    """The panel plus where it came from, so the terminal can say which path ran."""

    panel: Panel
    source: str            # "cache" | "built"
    paths: dict[str, str]
    warnings: list[str]


def get_panel(
    settings: Settings,
    use_cache: bool = True,
    lexicon: FrozenLexicon | None = None,
    force_default_lexicon: bool = False,
) -> PanelResult:
    """Return the panel for this dataset, from cache when the cache is provably current.

    The lexicon is resolved first because it is part of what the cache depends on: a panel scored
    with the shipped lexicon is a different table from one scored with an induced vocabulary, and
    the fingerprint check is what keeps the two from being confused for each other.
    """
    resolved = lexicon or resolve_lexicon(settings, force_default=force_default_lexicon)[0]
    if use_cache:
        cached = load_cached_panel(settings, expected_fingerprint=cache_fingerprint(settings, resolved.lexicon_id))
        if cached is not None and cached.meta:
            return PanelResult(
                panel=cached,
                source="cache",
                paths={},
                warnings=list(cached.meta.get("canonical_report", {}).get("warnings", [])),
            )

    canonical = build_canonical(settings)
    panel = build_panel(settings, canonical, resolved)
    paths = save_panel(panel, settings)
    return PanelResult(panel=panel, source="built", paths=paths, warnings=canonical.report["warnings"])
