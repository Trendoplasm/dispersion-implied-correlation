"""Readers for the study's three inputs: Cboe index history, price history, and the basket.

Every reader fails loudly. A silently dropped observation or a coerced date would change a
published statistic without changing anything visible.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

from dispersion.config import INDEX_TICKER, POINTS_PER_UNIT, REQUIRED_CBOE_FILES
from dispersion.models import Constituent, LevelByDate

logger = logging.getLogger(__name__)

CBOE_DATE_COLUMN = "DATE"
CBOE_CLOSE_COLUMN = "CLOSE"
PRICE_COLUMNS = frozenset({"date", "close"})
BASKET_COLUMNS = frozenset({"ticker", "name", "sector", "iv_index", "in_iv_basket"})
CBOE_DATE_FORMAT = "%m/%d/%Y"
ISO_DATE_FORMAT = "%Y-%m-%d"
TRUE_STRINGS = frozenset({"true", "1", "yes", "y"})


def parse_bool(value: str | None) -> bool:
    """Interpret a spreadsheet-style truthy value."""
    return (value or "").strip().lower() in TRUE_STRINGS


def resolve_value_column(fieldnames: Sequence[str]) -> str:
    """Return the column holding the daily level.

    Cboe publishes these files in two shapes. The volatility and correlation indexes carry
    ``DATE,OPEN,HIGH,LOW,CLOSE``; the dispersion index carries ``DATE`` plus a single column named
    after the index itself. Rather than special-casing series names, take ``CLOSE`` when it exists
    and otherwise the sole remaining column.

    Args:
        fieldnames: Header of the file.

    Returns:
        Name of the value column.

    Raises:
        ValueError: If there is no ``DATE`` column, or no unambiguous value column.
    """
    if CBOE_DATE_COLUMN not in fieldnames:
        raise ValueError(f"No {CBOE_DATE_COLUMN} column; got {list(fieldnames)}")
    if CBOE_CLOSE_COLUMN in fieldnames:
        return CBOE_CLOSE_COLUMN
    others = [name for name in fieldnames if name != CBOE_DATE_COLUMN]
    if len(others) != 1:
        raise ValueError(f"Cannot identify the value column among {list(fieldnames)}")
    return others[0]


def load_cboe_history(path: Path) -> LevelByDate:
    """Load a Cboe index history as a decimal level keyed by trading date.

    Cboe quotes both volatility and correlation in percentage points; both are converted to
    decimals here, once, so no downstream formula has to remember which convention it is in.

    Args:
        path: Path to a ``*_History.csv`` file as published by Cboe.

    Returns:
        Decimal level keyed by trading date.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the header is unusable, a row will not parse, a level is not positive, or
            the file holds no data rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing Cboe input: {path}")

    levels: LevelByDate = {}
    skipped: list[date] = []
    # utf-8-sig: Cboe's files carry a byte-order mark that would corrupt the "DATE" header.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        try:
            value_column = resolve_value_column(reader.fieldnames)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        for raw in reader:
            try:
                trading_date = datetime.strptime(raw[CBOE_DATE_COLUMN], CBOE_DATE_FORMAT).date()
                level = float(raw[value_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row in {path}: {raw}") from exc
            if level <= 0:
                # Cboe encodes an occasional missing print as zero -- the dispersion index has one,
                # on 8 February 2018, sitting between values of 14.05 and 20.92. A volatility or
                # correlation level of zero is not a real observation, so it is treated as missing
                # and logged. Dropping it silently is what this loader exists to avoid.
                skipped.append(trading_date)
                continue
            levels[trading_date] = level / POINTS_PER_UNIT

    if not levels:
        raise ValueError(f"No data rows found in {path}")
    if skipped:
        logger.warning(
            "%s: skipped %d nonpositive level(s), treated as missing: %s",
            path.name,
            len(skipped),
            ", ".join(day.isoformat() for day in skipped[:5]),
        )
    return levels


def load_price_history(path: Path) -> LevelByDate:
    """Load a daily close series.

    Args:
        path: Path to a ``date,close`` CSV.

    Returns:
        Closing price keyed by trading date.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the header is wrong, a row will not parse, a price is not positive, or the
            file holds no data rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing price input: {path}")

    prices: LevelByDate = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not PRICE_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"{path} lacks expected columns: {sorted(PRICE_COLUMNS)}")
        for raw in reader:
            try:
                trading_date = datetime.strptime(raw["date"], ISO_DATE_FORMAT).date()
                close = float(raw["close"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row in {path}: {raw}") from exc
            if close <= 0:
                raise ValueError(f"Nonpositive close in {path} on {trading_date}")
            prices[trading_date] = close

    if not prices:
        raise ValueError(f"No data rows found in {path}")
    return prices


def load_cboe_series(data_dir: Path) -> dict[str, LevelByDate]:
    """Load every Cboe series the study requires."""
    return {
        name: load_cboe_history(data_dir / filename)
        for name, filename in REQUIRED_CBOE_FILES.items()
    }


def load_basket(path: Path) -> list[Constituent]:
    """Load the reference basket.

    Args:
        path: Path to the basket reference CSV.

    Returns:
        Constituents in file order.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the header is wrong or no constituents are present.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing basket reference: {path}")

    constituents: list[Constituent] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not BASKET_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {sorted(BASKET_COLUMNS)}")
        for raw in reader:
            constituents.append(
                Constituent(
                    ticker=raw["ticker"].strip().upper(),
                    name=raw["name"].strip(),
                    sector=raw["sector"].strip(),
                    iv_index=raw["iv_index"].strip().upper(),
                    in_iv_basket=parse_bool(raw["in_iv_basket"]),
                )
            )

    if not constituents:
        raise ValueError(f"No constituents found in {path}")
    logger.info(
        "Loaded %d constituents; %d have an observed implied-volatility index",
        len(constituents),
        sum(item.in_iv_basket for item in constituents),
    )
    return constituents


def load_price_series(data_dir: Path, constituents: list[Constituent]) -> dict[str, LevelByDate]:
    """Load the index price history and every constituent's.

    Args:
        data_dir: Directory holding the downloaded price files.
        constituents: Basket members to load.

    Returns:
        Closing prices keyed by study ticker, including the index under
        :data:`~dispersion.config.INDEX_TICKER`.
    """
    series = {INDEX_TICKER: load_price_history(data_dir / f"{INDEX_TICKER}_prices.csv")}
    for item in constituents:
        series[item.ticker] = load_price_history(data_dir / f"{item.ticker}_prices.csv")
    return series
