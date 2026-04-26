"""Thin wrapper that delegates to the CLI's ``run-extended-experiments`` command."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_llm.cli import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "run-extended-experiments", *sys.argv[1:]]
    raise SystemExit(main())
