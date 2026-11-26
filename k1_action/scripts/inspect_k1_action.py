#!/usr/bin/env python
"""CLI alias: inspect a K1 action (same as validate_k1_action --stats)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from k1_action.scripts.validate_k1_action import main  # noqa: E402,F401

if __name__ == "__main__":
    if "--stats" not in sys.argv:
        sys.argv.append("--stats")
    main()
