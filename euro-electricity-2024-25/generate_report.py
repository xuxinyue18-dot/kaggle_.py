#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from euro_electricity_2024_25.pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
