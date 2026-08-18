"""Entry point for ``python -m dispersion``."""

from __future__ import annotations

import sys

from dispersion.cli import main

if __name__ == "__main__":
    sys.exit(main())
