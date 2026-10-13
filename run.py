"""Launcher for the demo. This is the file to run.

    .venv\\Scripts\\python.exe run.py doctor
    .venv\\Scripts\\python.exe run.py profile
    .venv\\Scripts\\python.exe run.py --dataset full profile

Why this exists: the package lives under src/ (src layout), and the project is not pip
installed, so `python -m eda_demo` would not find it. This adds src/ to the import path and
hands over to the real CLI. It is the only place that touches sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

# repo root == the directory holding this file; the package is at <root>/src/eda_demo
REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eda_demo.__main__ import main  # noqa: E402 - must come after the path is set up

if __name__ == "__main__":
    raise SystemExit(main())
