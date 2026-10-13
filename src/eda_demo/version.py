"""Code version — one constant, part of every `data_version`.

Why it is a separate module: SPEC.md section 4.8 defines `data_version` as
data hash + config hash + lexicon_id + code version. Keeping the code version in its own
file means a reviewer can see at a glance which release produced an artifact, and no module
has to import the CLI just to read a string.
"""

from __future__ import annotations

CODE_VERSION = "0.2.0"
