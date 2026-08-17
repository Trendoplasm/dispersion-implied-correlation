#!/usr/bin/env python3
"""Download the daily price histories the study needs.

Implied correlation is what the option market prices; realised correlation is what the stocks
actually did. Cboe supplies the implied side; this script supplies the realised side -- the index
plus every constituent in the reference basket.

Prices come from Yahoo Finance's public chart endpoint. It needs no API key, which is why it is
used here, but it is an undocumented endpoint and can change without notice. If it fails, the same
data can be exported by hand from any market-data provider into the same simple CSV shape
(`date,close`) and the study will read it unchanged.

    python scripts/fetch_price_data.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

#: The index whose implied correlation is observed.
INDEX_SYMBOL = "^GSPC"

#: Path to the reference basket, whose constituents are read at runtime rather than hardcoded, so
#: the basket is version-controlled data rather than something buried in a script.
BASKET_PATH = Path("data/reference/basket.csv")


def read_basket(path: Path) -> list[str]:
    """Return the basket's constituent tickers.

    Raises:
        RuntimeError: If the reference file is missing or has no usable rows.
    """
    if not path.exists():
        raise RuntimeError(f"Missing basket reference: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        tickers = [row["ticker"].strip() for row in csv.DictReader(handle) if row.get("ticker")]
    if not tickers:
        raise RuntimeError(f"No constituents listed in {path}")
    return tickers


#: Twenty years covers the 2011 start of the single-name volatility indexes with room to spare.
RANGE = "20y"
TIMEOUT_SECONDS = 60
USER_AGENT = "Mozilla/5.0 (compatible; volatility-risk-premium research script)"
DEFAULT_DEST = Path("data/raw")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download daily underlying price histories.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Download directory.")
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument("--basket", type=Path, default=BASKET_PATH, help="Reference basket CSV.")
    return parser.parse_args(argv)


def download_prices(symbol: str) -> list[tuple[str, float]]:
    """Fetch split-adjusted daily closes for one symbol.

    Args:
        symbol: Yahoo Finance symbol.

    Returns:
        Ascending ``(iso date, close)`` pairs, with non-trading placeholders removed.

    Raises:
        RuntimeError: If the request fails or the response is not the expected shape.
    """
    query = urllib.parse.urlencode({"range": RANGE, "interval": "1d"})
    url = f"{CHART_URL.format(symbol=urllib.parse.quote(symbol))}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not download {symbol}: {exc}") from exc

    try:
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        # "close" is split-adjusted but not dividend-adjusted, which is the price path an option
        # actually references. Dividends enter the model separately, as a carry yield.
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response shape for {symbol}: {exc}") from exc

    rows = [
        (datetime.fromtimestamp(timestamp, tz=UTC).date().isoformat(), float(close))
        for timestamp, close in zip(timestamps, closes, strict=True)
        if close is not None
    ]
    if not rows:
        raise RuntimeError(f"No usable observations returned for {symbol}")
    return rows


def write_prices(path: Path, rows: list[tuple[str, float]]) -> None:
    """Write price rows as a two-column CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "close"])
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        constituents = read_basket(args.basket)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    symbols = {"INDEX": INDEX_SYMBOL} | {ticker: ticker for ticker in constituents}
    print(f"Fetching {len(symbols)} price histories into {args.dest}")
    written = 0
    for name, symbol in symbols.items():
        target = args.dest / f"{name}_prices.csv"
        if target.exists() and not args.force:
            print(f"  {target.name}: already present, skipping")
            continue
        try:
            rows = download_prices(symbol)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        write_prices(target, rows)
        written += 1
        print(f"  {target.name}: {len(rows)} observations, {rows[0][0]} to {rows[-1][0]}")
    print(f"Done: {written} written, {len(symbols) - written} already present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
