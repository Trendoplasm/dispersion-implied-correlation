#!/usr/bin/env python3
"""Download the Cboe index histories this study uses.

Cboe's data is published under Cboe's terms of use and is not redistributed with this
repository, so a fresh clone fetches it from the source:

    python scripts/fetch_cboe_data.py

Three groups of series are needed:

* **Implied correlation** (COR1M / COR3M / COR6M) -- Cboe's S&P 500 Implied Correlation Indexes.
  This is the study's central quantity, *observed* rather than inferred: it is the average
  correlation the option market prices between S&P 500 constituents.
* **Dispersion** (DSPX) -- Cboe's S&P 500 Dispersion Index, an independent observed measure of
  the same phenomenon, used as a cross-check.
* **Volatility** (VIX plus five single-name indexes) -- the index and constituent volatility
  anchors the basket identity needs.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices"

#: Series required by the study, with the role each one plays.
SERIES: dict[str, str] = {
    "COR1M": "S&P 500 1-month implied correlation (the observed premium's implied leg)",
    "COR3M": "S&P 500 3-month implied correlation",
    "COR6M": "S&P 500 6-month implied correlation",
    "DSPX": "S&P 500 Dispersion Index (independent cross-check)",
    "VIX": "S&P 500 30-day expected volatility (the index leg of the basket identity)",
    "VXAPL": "Apple 30-day expected volatility",
    "VXAZN": "Amazon 30-day expected volatility",
    "VXGOG": "Alphabet 30-day expected volatility",
    "VXGS": "Goldman Sachs 30-day expected volatility",
    "VXIBM": "IBM 30-day expected volatility",
}

DEFAULT_DEST = Path("data/raw")
TIMEOUT_SECONDS = 60


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Cboe index histories.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Download directory.")
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    return parser.parse_args(argv)


def fetch(series: str, dest: Path, *, force: bool) -> bool:
    """Download one series, returning True if a file was written."""
    filename = f"{series}_History.csv"
    target = dest / filename
    if target.exists() and not force:
        print(f"  {filename}: already present, skipping")
        return False

    url = f"{BASE_URL}/{filename}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc

    if b"DATE" not in payload[:200].upper():
        raise RuntimeError(f"{url} did not return a Cboe history file")

    dest.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    print(f"  {filename}: {len(payload) / 1024:.0f} KiB written")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Fetching {len(SERIES)} Cboe index histories into {args.dest}")
    try:
        written = sum(fetch(series, args.dest, force=args.force) for series in SERIES)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Done: {written} written, {len(SERIES) - written} already present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
