"""Raw file loading and input hashing.

Design decision embodied here: inputs are opened READ-ONLY and, for a zipped source, the
CSV member is read straight out of the archive in memory. Nothing is ever extracted into
`data/raw/`, so the delivered files stay byte-identical to what was received and a hash
taken before a run still matches after it.

A second decision: every column is read as `str` with NA-coercion disabled. Type inference
at load time would turn an unparseable date into NaT and an empty cell into NaN *silently*,
and a silently-dropped row changes the dataset without changing any reported count. Parsing
is therefore an explicit, loud step in later modules.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pandas as pd

# 1 MiB read blocks: large enough to be fast, small enough that a 6 MB zip never loads whole
# just to be hashed. Used for both container and member hashing.
_HASH_BLOCK = 1 << 20


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in blocks so it is never loaded whole."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 of an in-memory payload (used for a zip member)."""
    return hashlib.sha256(payload).hexdigest()


def _resolve_member(archive: zipfile.ZipFile, path: Path, member: str | None) -> str:
    """Return the CSV member name to read, or fail with the available names listed.

    Prefers the configured member name; falls back to the only CSV in the archive. An
    archive with several CSVs and no configured member is an error rather than a guess,
    because guessing would silently analyse the wrong file.
    """
    names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
    if member is not None:
        if member not in names:
            raise SystemExit(f"'{member}' is not a CSV member of {path.name} (found: {names})")
        return member
    if len(names) == 1:
        return names[0]
    raise SystemExit(f"{path.name} holds {len(names)} CSVs; set tweets_member in settings.yaml: {names}")


def read_raw_csv(path: Path, member: str | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read a CSV — or the named member of a ZIP — as all-string columns.

    Returns (dataframe, hashes) where hashes holds 'container' and, for a zip, 'member'.
    Never extracts to disk and never modifies the source.

    `encoding="utf-8-sig"` is deliberate: it drops a byte-order mark if one is present and
    is identical to utf-8 otherwise. The 1:9 split files were written with a BOM, so
    reading them as plain utf-8 would name the first column "\\ufeffDate" and every
    downstream column lookup would fail.
    """
    if not path.exists():
        raise SystemExit(f"missing input file: {path}")

    hashes = {"container": sha256_file(path)}
    read_kwargs = {"dtype": str, "keep_default_na": False, "na_filter": False, "encoding": "utf-8-sig"}

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            target = _resolve_member(archive, path, member)
            payload = archive.read(target)
        hashes["member"] = sha256_bytes(payload)
        hashes["member_name"] = target
        # BytesIO keeps the member in memory; nothing is written to disk.
        from io import BytesIO

        return pd.read_csv(BytesIO(payload), **read_kwargs), hashes

    return pd.read_csv(path, **read_kwargs), hashes
