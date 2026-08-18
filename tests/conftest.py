"""Synthetic fixtures with analytically exact properties.

Tests build their own return series rather than reading market data, so the suite runs anywhere.

The construction is worth understanding, because it makes correlation tests *exact* rather than
approximate. Rows of a Hadamard matrix are mutually orthogonal, have zero mean, and share the same
norm. Taking one row as a common factor and a distinct row as each constituent's idiosyncratic
term,

    r_i = sqrt(rho) * factor + sqrt(1 - rho) * idiosyncratic_i

gives every pair a correlation of exactly ``rho`` -- no sampling error at all. So a test can assert
that a measured correlation equals 0.4 rather than merely lies near it.

One caveat, and the tests respect it: orthogonality is a property of the *whole* matrix, not of an
arbitrary contiguous slice of it. Measured over the full series the correlation is exact; measured
over a 60-day sub-window it lands within a few thousandths. Tests that measure the full series
assert equality; tests that measure a rolling window assert a stated tolerance.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import hadamard

from dispersion.config import (
    CONSTITUENT_VOLATILITY_INDEXES,
    INDEX_TICKER,
    INDEX_VOLATILITY_INDEX,
    CostModel,
    StudyConfig,
)
from dispersion.models import Constituent, LevelByDate

#: Observations per synthetic window. A power of two, as the Hadamard construction requires.
OBSERVATIONS = 256

#: Correlation planted in the standard fixture.
KNOWN_CORRELATION = 0.40

#: How far a rolling-window correlation can sit from the planted value, given that the orthogonal
#: construction is exact only over the full series. The deviation grows as the window shortens:
#: under 0.002 over 60 observations, under 0.04 over 21.
WINDOW_TOLERANCE = 5e-3
SHORT_WINDOW_TOLERANCE = 5e-2

#: Constituents in the standard fixture, comfortably above the study's minimum.
CONSTITUENT_COUNT = 24

#: Daily return scale, chosen so annualised volatility is a round number.
DAILY_SCALE = 0.20 / math.sqrt(252.0)

START_PRICE = 100.0

#: Tickers that carry an option leg. Derived from the study's configuration rather than repeated,
#: so a fixture can never disagree with the real index names (Cboe calls Apple's index VXAPL, not
#: VXAAPL, and Alphabet's VXGOG, not VXGOOGL).
TRADEABLE = tuple(CONSTITUENT_VOLATILITY_INDEXES)


def correlated_returns(
    count: int = CONSTITUENT_COUNT,
    correlation: float = KNOWN_CORRELATION,
    observations: int = OBSERVATIONS,
    scale: float = DAILY_SCALE,
) -> np.ndarray:
    """Build a return matrix whose every pairwise correlation is exactly ``correlation``.

    Args:
        count: Number of constituents.
        correlation: Pairwise correlation to plant, in [0, 1].
        observations: Returns per constituent; must be a power of two.
        scale: Standard deviation of each constituent's returns.

    Returns:
        Constituents by observations.

    Raises:
        ValueError: If the matrix is too small to supply orthogonal rows.
    """
    basis = hadamard(observations).astype(float)
    if count + 1 > observations - 1:
        raise ValueError("Not enough orthogonal rows for the requested constituent count")
    # Row 0 of a Hadamard matrix is all ones, so it is skipped: a constant row has no variance.
    factor = basis[1]
    idiosyncratic = basis[2 : 2 + count]
    weight = math.sqrt(correlation)
    residual = math.sqrt(1.0 - correlation)
    combined = weight * factor + residual * idiosyncratic
    # Every row now has the same norm; rescale so each has the requested standard deviation.
    scaled = combined / combined.std(axis=1, ddof=1, keepdims=True) * scale
    return np.asarray(scaled, dtype=float)


def prices_from_returns(returns: np.ndarray, start: float = START_PRICE) -> np.ndarray:
    """Convert log returns into price paths beginning at ``start``."""
    steps = np.concatenate([np.zeros((returns.shape[0], 1)), returns.cumsum(axis=1)], axis=1)
    return np.asarray(start * np.exp(steps), dtype=float)


def trading_dates(start: date, count: int) -> list[date]:
    """Generate ascending weekday dates."""
    dates: list[date] = []
    current = start
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def constituent(ticker: str, *, tradeable: bool = False) -> Constituent:
    """Return one basket member."""
    return Constituent(
        ticker=ticker,
        name=f"{ticker} test security",
        sector="Test",
        iv_index=f"VX{ticker}" if tradeable else "",
        in_iv_basket=tradeable,
    )


def synthetic_basket(count: int = CONSTITUENT_COUNT) -> list[Constituent]:
    """Return a basket whose first five members carry option legs."""
    tickers = [*TRADEABLE, *[f"N{index:02d}" for index in range(count - len(TRADEABLE))]]
    return [constituent(t, tradeable=t in TRADEABLE) for t in tickers[:count]]


def synthetic_prices(
    basket: Sequence[Constituent], dates: Sequence[date], correlation: float = KNOWN_CORRELATION
) -> dict[str, LevelByDate]:
    """Build index and constituent price histories with a known correlation.

    The index is the equal-weighted basket, so the variance identity holds exactly between it and
    its members rather than approximately.
    """
    returns = correlated_returns(len(basket), correlation, observations=OBSERVATIONS)
    needed = len(dates) - 1
    if needed > returns.shape[1]:
        raise ValueError("Not enough synthetic returns for the requested calendar")
    returns = returns[:, :needed]
    paths = prices_from_returns(returns)

    series: dict[str, LevelByDate] = {}
    for row, item in enumerate(basket):
        series[item.ticker] = dict(zip(dates, paths[row], strict=True))
    index_returns = returns.mean(axis=0)
    index_path = START_PRICE * np.exp(np.concatenate([[0.0], index_returns.cumsum()]))
    series["INDEX"] = dict(zip(dates, index_path, strict=True))
    return series


def flat_series(dates: Sequence[date], level: float) -> LevelByDate:
    """Return a constant level history."""
    return dict.fromkeys(dates, level)


def cboe_series(
    dates: Sequence[date],
    *,
    implied_correlation: float = 0.45,
    index_iv: float = 0.18,
    constituent_iv: float = 0.28,
) -> dict[str, LevelByDate]:
    """Build a full set of Cboe series at constant levels."""
    series = {
        "COR1M": flat_series(dates, implied_correlation),
        "COR3M": flat_series(dates, implied_correlation + 0.04),
        "COR6M": flat_series(dates, implied_correlation + 0.07),
        "DSPX": flat_series(dates, 1.0 - implied_correlation),
        INDEX_VOLATILITY_INDEX: flat_series(dates, index_iv),
    }
    for index_name in CONSTITUENT_VOLATILITY_INDEXES.values():
        series[index_name] = flat_series(dates, constituent_iv)
    return series


def volatility_map(
    dates: Sequence[date], *, index_iv: float = 0.18, constituent_iv: float = 0.28
) -> dict[str, LevelByDate]:
    """Return implied volatility keyed by *ticker*, as the trade simulation expects it."""
    series = cboe_series(dates, index_iv=index_iv, constituent_iv=constituent_iv)
    return {INDEX_TICKER: series[INDEX_VOLATILITY_INDEX]} | {
        ticker: series[index_name] for ticker, index_name in CONSTITUENT_VOLATILITY_INDEXES.items()
    }


def write_cboe_csv(
    path: Path, dates: Sequence[date], levels: Sequence[float], *, value_column: str = "CLOSE"
) -> None:
    """Write a Cboe-format history file in either published shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        if value_column == "CLOSE":
            writer.writerow(["DATE", "OPEN", "HIGH", "LOW", "CLOSE"])
            for day, level in zip(dates, levels, strict=True):
                writer.writerow([day.strftime("%m/%d/%Y"), level, level, level, level])
        else:
            writer.writerow(["DATE", value_column])
            for day, level in zip(dates, levels, strict=True):
                writer.writerow([day.strftime("%m/%d/%Y"), level])


def write_price_csv(path: Path, series: LevelByDate) -> None:
    """Write a two-column price history file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "close"])
        for day in sorted(series):
            writer.writerow([day.isoformat(), series[day]])


def write_basket_csv(path: Path, basket: Sequence[Constituent]) -> None:
    """Write a basket reference file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["ticker", "name", "sector", "iv_index", "in_iv_basket"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in basket:
            writer.writerow(
                {
                    "ticker": item.ticker,
                    "name": item.name,
                    "sector": item.sector,
                    "iv_index": item.iv_index,
                    "in_iv_basket": "TRUE" if item.in_iv_basket else "FALSE",
                }
            )


@pytest.fixture
def dates() -> list[date]:
    """Return 250 weekday trading dates from 2013."""
    return trading_dates(date(2013, 1, 2), 250)


@pytest.fixture
def basket() -> list[Constituent]:
    """Return the standard synthetic basket."""
    return synthetic_basket()


@pytest.fixture
def prices(basket: list[Constituent], dates: list[date]) -> dict[str, LevelByDate]:
    """Return synthetic price histories with a known pairwise correlation."""
    return synthetic_prices(basket, dates)


@pytest.fixture
def cboe(dates: list[date]) -> dict[str, LevelByDate]:
    """Return synthetic Cboe series at constant levels."""
    return cboe_series(dates)


@pytest.fixture
def config() -> StudyConfig:
    """Return a configuration sized for the synthetic calendar."""
    return StudyConfig(
        start_date="2013-01-02",
        end_date="2026-06-30",
        realized_window_days=60,
        zscore_window_days=100,
        warmup_days=70,
        bootstrap_iterations=200,
        train_end="2013-06-30",
    )


@pytest.fixture
def costs() -> CostModel:
    """Return the default cost model."""
    return CostModel()


@pytest.fixture
def free_costs() -> CostModel:
    """Return a cost model with every cost switched off."""
    return CostModel(
        option_commission=0.0,
        option_half_spread=0.0,
        index_hedge_cost_bps=0.0,
        stock_hedge_cost_bps=0.0,
    )


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parent.parent
