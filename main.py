"""Project entry point.  Adds ``src/`` to sys.path and delegates to :func:`cli.main`."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_llm.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
