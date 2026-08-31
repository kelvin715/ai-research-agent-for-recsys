#!/usr/bin/env python3
"""Launch the read-only live experiment graph dashboard."""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "orchestrator") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "orchestrator"))

from graph_view import main  # noqa: E402


if __name__ == "__main__":
    main()
