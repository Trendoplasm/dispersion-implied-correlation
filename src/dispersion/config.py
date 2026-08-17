"""Study parameters and the fixed data contract.

Values here are research choices carried over from the original study's Assumptions sheet, or
external constraints on what data exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Final

# --- External data contract -------------------------------------------------------------

#: Cboe implied-correlation indexes, by nominal horizon in calendar days. These are the study's
#: central quantity and they are *observed*, not inferred: Cboe publishes the average correlation
#: the option market prices between S&P 500 constituents.
IMPLIED_CORRELATION_INDEXES: Final[dict[str, float]] = {
    "COR1M": 30.0,
    "COR3M": 93.0,
    "COR6M": 186.0,
}

#: Cboe S&P 500 Dispersion Index -- an independently constructed observed measure of the same
#: phenomenon, used to cross-check the correlation series rather than as an input.
DISPERSION_INDEX: Final[str] = "DSPX"

#: 30-day expected volatility of the index whose options form the short leg of the trade.
INDEX_VOLATILITY_INDEX: Final[str] = "VIX"

#: Constituents for which Cboe publishes a 30-day volatility index. These five are the only
#: single names whose option legs can be priced from observed implied volatility, which is what
#: bounds the traded basket.
CONSTITUENT_VOLATILITY_INDEXES: Final[dict[str, str]] = {
    "AAPL": "VXAPL",
    "AMZN": "VXAZN",
    "GOOGL": "VXGOG",
    "GS": "VXGS",
    "IBM": "VXIBM",
}

#: Every Cboe series the study downloads.
REQUIRED_CBOE_FILES: Final[dict[str, str]] = {
    name: f"{name}_History.csv"
    for name in (
        *IMPLIED_CORRELATION_INDEXES,
        DISPERSION_INDEX,
        INDEX_VOLATILITY_INDEX,
        *CONSTITUENT_VOLATILITY_INDEXES.values(),
    )
}

#: Study ticker of the index.
INDEX_TICKER: Final[str] = "INDEX"

#: Reporting order for pooled and per-regime summaries.
REGIME_ORDER: Final[tuple[str, ...]] = ("Low volatility", "Middle volatility", "High volatility")

# --- Conventions ------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: Final[float] = 252.0
CALENDAR_DAYS_PER_YEAR: Final[float] = 365.0

#: Cboe quotes both volatility and correlation in percentage points; the models work in decimals.
POINTS_PER_UNIT: Final[float] = 100.0

#: Minimum constituents with data on a date before a basket correlation is reported. Below this the
#: estimate says more about which names were missing than about the market.
MIN_CONSTITUENTS: Final[int] = 20

#: Minimum returns in a window before a correlation or volatility estimate is reported.
MIN_WINDOW_OBSERVATIONS: Final[int] = 15

#: Quantile bounds of a two-sided 95% bootstrap interval.
BOOTSTRAP_LOWER_QUANTILE: Final[float] = 0.025
BOOTSTRAP_UPPER_QUANTILE: Final[float] = 0.975

#: Number of trades listed in the best- and worst-outcome tables.
EXTREME_TRADE_COUNT: Final[int] = 15


@dataclass(frozen=True)
class CostModel:
    """Transaction costs applied to the dispersion structure.

    Attributes:
        option_commission: Dollars per contract per leg.
        option_half_spread: Entry and exit half-spread as a fraction of option premium.
        index_hedge_cost_bps: Basis points of index turnover when delta-hedging.
        stock_hedge_cost_bps: Basis points of single-name turnover when delta-hedging.
        contract_multiplier: Shares per option contract.
    """

    option_commission: float = 0.65
    option_half_spread: float = 0.01
    index_hedge_cost_bps: float = 0.5
    stock_hedge_cost_bps: float = 1.0
    contract_multiplier: float = 100.0

    def hedge_cost_bps(self, ticker: str) -> float:
        """Return the hedge cost in basis points for one security."""
        return self.index_hedge_cost_bps if ticker == INDEX_TICKER else self.stock_hedge_cost_bps


@dataclass(frozen=True)
class StudyConfig:
    """Windows, thresholds, and inference settings for one run.

    Attributes:
        start_date: First date of the study period.
        end_date: Last date of the study period. Frozen deliberately: the data providers extend
            their series every trading day, so an open-ended sample would answer differently on
            every download. Fixing the end is what makes published results reproducible later.
        risk_free_rate: Annual continuously compounded financing rate.
        index_dividend_yield: Continuous dividend yield of the index.
        constituent_dividend_yield: Continuous dividend yield applied to single names.
        holding_days: Trading days from entry to expiry, and the spacing between entries.
        realized_window_days: Trailing window for the realised-correlation leg of the signal.
        zscore_window_days: Trailing window for the signal's mean and standard deviation.
        warmup_days: Trading days before the first trade, so the z-score window is populated.
        short_correlation_z: Z-score above which the study sells index volatility.
        long_correlation_z: Z-score below which the study buys index volatility.
        target_gross_vega: Dollar vega each leg of the structure is scaled to.
        short_option_margin_fraction: Research capital proxy as a fraction of short notional.
        train_end: Last date of the period used to choose thresholds.
        bootstrap_iterations: Resamples used for confidence intervals.
        random_seed: Seed for the bootstrap generator.
    """

    start_date: str = "2013-01-02"
    end_date: str = "2026-06-30"
    risk_free_rate: float = 0.0425
    index_dividend_yield: float = 0.013
    constituent_dividend_yield: float = 0.008
    holding_days: int = 21
    realized_window_days: int = 60
    zscore_window_days: int = 252
    warmup_days: int = 378
    short_correlation_z: float = 0.75
    long_correlation_z: float = -0.90
    target_gross_vega: float = 100_000.0
    short_option_margin_fraction: float = 0.20
    train_end: str = "2020-12-31"
    bootstrap_iterations: int = 10_000
    random_seed: int = 20_260_819

    def start(self) -> date:
        """Return :attr:`start_date` as a date."""
        return datetime.strptime(self.start_date, "%Y-%m-%d").date()

    def end(self) -> date:
        """Return :attr:`end_date` as a date."""
        return datetime.strptime(self.end_date, "%Y-%m-%d").date()

    def train_cutoff(self) -> date:
        """Return :attr:`train_end` as a date."""
        return datetime.strptime(self.train_end, "%Y-%m-%d").date()

    def horizon_years(self, trading_days: int) -> float:
        """Convert a trading-day horizon to a year fraction."""
        return trading_days / TRADING_DAYS_PER_YEAR
