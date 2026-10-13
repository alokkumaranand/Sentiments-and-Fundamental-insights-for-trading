"""Configuration loading — the ONLY module that reads `.env` or the YAML config files.

Why this exists as one module: SPEC.md section 5.1 fixes a precedence order
(process environment > .env > config/settings.yaml > code defaults). Scattering that
logic across the codebase is how a secret ends up in a log or a run stops reproducing.
Every other module receives a resolved `Settings` object instead of reading config itself.

Nothing here touches the network or the dataset; it only reads two YAML files, one
optional .env, and a handful of environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# src/eda_demo/config.py -> parents[0]=eda_demo, [1]=src, [2]=repository root.
# Every configured path is resolved against this, so a run works from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]

SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"
EXPERIMENTS_PATH = REPO_ROOT / "config" / "experiments.yaml"
ENV_PATH = REPO_ROOT / ".env"


@dataclass(frozen=True)
class Settings:
    """Fully resolved configuration for one run.

    `tweets_member` is None for a plain CSV and names the member inside the zip for a
    zipped source. Keeping it here means the loader never has to guess or re-parse config.
    """

    repo_root: Path
    raw: dict[str, Any]           # the whole settings.yaml, for values not lifted out below
    experiments: dict[str, Any]   # the whole experiments.yaml
    dataset_name: str
    tweets_path: Path
    prices_path: Path
    tweets_member: str | None
    artifacts_dir: Path
    quality_report_path: Path
    market_timezone: str
    close_hour: int

    def env(self, key: str, default: str | None = None) -> str | None:
        """Read an environment variable by the name given in settings.yaml.

        Returns `default` when the variable is unset or empty. This indirection is why
        settings.yaml can say `base_url_env: EDA_LLM_BASE_URL` without hard-coding the
        variable name in code.
        """
        return os.environ.get(key) or default

    def section(self, name: str) -> dict[str, Any]:
        """Return one top-level settings.yaml section as a dict ({} when the section is absent).

        A missing section returns an empty dict rather than raising, because every consumer
        supplies its own default; that keeps the config file able to grow a key at a time.
        """
        return self.raw.get(name, {}) or {}

    def agent(self, name: str) -> dict[str, Any]:
        """Return one agent's budget block from settings.yaml -> agents.<name>."""
        return (self.section("agents")).get(name, {}) or {}


def _read_yaml(path: Path) -> dict[str, Any]:
    """Return a YAML file as a dict, failing loudly with the path in the message.

    An empty file yields {} rather than None, so callers never have to null-check the
    result of a config read.
    """
    if not path.exists():
        raise SystemExit(f"missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def _resolve(relative: str | Path) -> Path:
    """Resolve a configured path against the repository root (absolute paths pass through)."""
    candidate = Path(relative)
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


def load_settings(dataset: str | None = None) -> Settings:
    """Load and resolve configuration. `dataset` overrides `datasets.active` for one run.

    Raises SystemExit with the list of available names when an unknown dataset is asked
    for, because a typo in a dataset name should stop the run, not silently fall back to
    the default and produce a report about the wrong data.
    """
    # load_dotenv does not overwrite variables already present in the process, which is
    # exactly the precedence we want: a real environment variable beats .env.
    load_dotenv(ENV_PATH, override=False)

    raw = _read_yaml(SETTINGS_PATH)
    experiments = _read_yaml(EXPERIMENTS_PATH)

    sources = raw.get("datasets", {}).get("sources", {})
    if not sources:
        raise SystemExit(f"no datasets.sources defined in {SETTINGS_PATH}")

    name = dataset or raw.get("datasets", {}).get("active")
    if name not in sources:
        raise SystemExit(f"unknown dataset '{name}'. Available: {sorted(sources)}")

    entry = sources[name] or {}
    for key in ("tweets", "prices"):
        if not entry.get(key):
            raise SystemExit(f"dataset '{name}' has no '{key}' path in {SETTINGS_PATH}")

    market = raw.get("market", {})
    paths = raw.get("paths", {})
    return Settings(
        repo_root=REPO_ROOT,
        raw=raw,
        experiments=experiments,
        dataset_name=name,
        tweets_path=_resolve(entry["tweets"]),
        prices_path=_resolve(entry["prices"]),
        tweets_member=entry.get("tweets_member") or None,
        artifacts_dir=_resolve(paths.get("artifacts", "artifacts/runs")),
        quality_report_path=_resolve(paths.get("quality_report", "docs/quality_report.md")),
        market_timezone=str(market.get("timezone", "America/New_York")),
        close_hour=int(market.get("close_hour", 16)),
    )
