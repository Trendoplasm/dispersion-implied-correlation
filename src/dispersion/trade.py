"""The dispersion structure: index options against constituent options.

A dispersion trade takes a view on correlation without taking a view on the market's direction. It
sells volatility on the index and buys it on the constituents, or the reverse. Because index
variance is constituent variance plus a correlation term, what remains after the two legs offset
is exposure to how much the constituents actually moved *together*.

Sizing
------
Both sides are scaled to the same dollar vega, so a parallel shift in every implied volatility
nets out and only the *spread* between index and constituent volatility drives profit. That is
what makes it a correlation trade rather than a disguised short-volatility trade.

Every leg is a straddle, marked daily with its own observed implied-volatility index and its own
observed close, delta-hedged in its own underlying, and held to cash settlement.

Attribution
-----------
Profit splits into three pieces that sum exactly to the total:

* **Correlation P&L** -- vega multiplied by how each leg's implied volatility moved *relative to*
  the vega-weighted average move. This is the dispersion of volatility changes, which is the
  trade's thesis.
* **Volatility P&L** -- net vega multiplied by the average move. Near zero by construction, and
  reporting it is how you check the structure really was vega-neutral.
* **Residual** -- realised gamma, time decay, hedging error, financing. Everything the first-order
  volatility terms do not explain.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from dispersion.blackscholes import StraddleMark, forward_price, straddle_mark
from dispersion.config import (
    INDEX_TICKER,
    TRADING_DAYS_PER_YEAR,
    CostModel,
    StudyConfig,
)
from dispersion.models import LevelByDate, OptionLeg, Row
from dispersion.signals import LONG_CORRELATION, SHORT_CORRELATION

logger = logging.getLogger(__name__)

DAY_FRACTION = 1.0 / TRADING_DAYS_PER_YEAR
BASIS_POINTS = 10_000.0


@dataclass(frozen=True)
class DispersionTrade:
    """One completed dispersion trade.

    Attributes:
        entry_date: Date the structure was opened.
        exit_date: Date it settled.
        direction: ``short_correlation`` or ``long_correlation``.
        entry_zscore: Standardised signal at entry.
        entry_implied_correlation: Observed implied correlation at entry.
        realized_correlation: Average pairwise correlation actually delivered over the holding
            period.
        correlation_change: Realised minus implied. Negative favours a short-correlation trade.
        index_entry_iv: Index implied volatility at entry.
        mean_constituent_entry_iv: Vega-weighted mean constituent implied volatility at entry.
        index_iv_change: Change in index implied volatility over the trade.
        mean_constituent_iv_change: Vega-weighted mean change in constituent implied volatility.
        legs: Every option leg as entered.
        gross_vega: Dollar vega on each side at entry.
        net_pnl: Profit after costs, in dollars.
        correlation_pnl: Portion attributed to the dispersion of volatility changes.
        volatility_pnl: Portion attributed to the average volatility change.
        residual_pnl: Portion the first-order terms do not explain.
        index_pnl: Profit contributed by the index leg, before costs.
        constituent_pnl: Profit contributed by the constituent legs, before costs.
        cost: Transaction costs in dollars.
        capital: Research capital proxy in dollars.
        return_on_capital: :attr:`net_pnl` divided by :attr:`capital`.
        rebalances: Delta rebalances performed across all legs.
        in_sample: Whether entry falls in the threshold-selection period.
    """

    entry_date: date
    exit_date: date
    direction: str
    entry_zscore: float
    entry_implied_correlation: float
    realized_correlation: float | None
    correlation_change: float | None
    index_entry_iv: float
    mean_constituent_entry_iv: float
    index_iv_change: float
    mean_constituent_iv_change: float
    legs: tuple[OptionLeg, ...]
    gross_vega: float
    net_pnl: float
    correlation_pnl: float
    volatility_pnl: float
    residual_pnl: float
    index_pnl: float
    constituent_pnl: float
    cost: float
    capital: float
    return_on_capital: float
    rebalances: int
    in_sample: bool


class _LegState:
    """Mutable state of one option leg while the trade is open."""

    def __init__(
        self,
        ticker: str,
        quantity: float,
        strike: float,
        spot: float,
        implied_volatility: float,
        mark: StraddleMark,
        dividend_yield: float,
    ) -> None:
        self.ticker = ticker
        self.quantity = quantity
        self.strike = strike
        self.dividend_yield = dividend_yield
        self.spot = spot
        self.implied_volatility = implied_volatility
        self.mark = mark
        self.entry_premium = mark.value
        # A short straddle has negative delta exposure to hedge, and vice versa.
        self.hedge_shares = -quantity * mark.delta
        self.pnl = 0.0

    @property
    def dollar_vega(self) -> float:
        """Signed dollar vega per one unit of volatility, at the current mark."""
        return self.quantity * self.mark.vega * 100.0


def _leg_quantity(
    target_vega: float, mark: StraddleMark, contract_multiplier: float, sign: float
) -> float:
    """Return the straddle quantity that puts a leg at the target dollar vega."""
    vega_per_straddle = mark.vega * contract_multiplier
    if vega_per_straddle <= 0:
        return 0.0
    return sign * target_vega / vega_per_straddle


def simulate_trade(
    dates: Sequence[date],
    entry_index: int,
    direction: str,
    entry_zscore: float,
    entry_implied_correlation: float,
    realized_correlation: float | None,
    price_series: Mapping[str, LevelByDate],
    volatility_series: Mapping[str, LevelByDate],
    constituent_tickers: Sequence[str],
    config: StudyConfig,
    costs: CostModel,
) -> DispersionTrade | None:
    """Simulate one dispersion trade from entry to settlement.

    Args:
        dates: Trading calendar.
        entry_index: Position in ``dates`` at which to open.
        direction: ``short_correlation`` or ``long_correlation``.
        entry_zscore: Standardised signal at entry.
        entry_implied_correlation: Observed implied correlation at entry.
        realized_correlation: Correlation delivered over the holding period, if measurable.
        price_series: Closing prices keyed by ticker, including the index.
        volatility_series: Implied-volatility series keyed by ticker, including the index.
        constituent_tickers: Names eligible to carry a constituent leg.
        config: Sizing, rates, and holding period.
        costs: Transaction-cost model.

    Returns:
        The completed trade, or None when the window runs past the end of the data or too little
        of the structure can be priced at entry.
    """
    exit_index = entry_index + config.holding_days
    if exit_index >= len(dates):
        return None

    entry_date = dates[entry_index]
    maturity = config.horizon_years(config.holding_days)
    rate = config.risk_free_rate
    index_sign = -1.0 if direction == SHORT_CORRELATION else 1.0
    constituent_sign = -index_sign

    index_spot = price_series[INDEX_TICKER].get(entry_date)
    index_iv = volatility_series[INDEX_TICKER].get(entry_date)
    if index_spot is None or index_iv is None:
        return None

    available = [
        ticker
        for ticker in constituent_tickers
        if price_series[ticker].get(entry_date) is not None
        and volatility_series[ticker].get(entry_date) is not None
    ]
    if not available:
        return None
    weight = 1.0 / len(available)

    states: list[_LegState] = []
    legs: list[OptionLeg] = []
    cost = 0.0

    def open_leg(
        ticker: str, spot: float, implied: float, target: float, sign: float, yield_: float
    ) -> None:
        nonlocal cost
        strike = float(forward_price(spot, rate, yield_, maturity))
        mark = straddle_mark(spot, strike, rate, yield_, implied, maturity)
        quantity = _leg_quantity(target, mark, costs.contract_multiplier, sign)
        if quantity == 0.0:
            return
        state = _LegState(ticker, quantity, strike, spot, implied, mark, yield_)
        states.append(state)
        contracts = abs(quantity)
        cost += costs.option_half_spread * mark.value * contracts * costs.contract_multiplier
        cost += 2.0 * costs.option_commission * contracts
        cost += abs(state.hedge_shares) * spot * costs.hedge_cost_bps(ticker) / BASIS_POINTS
        legs.append(
            OptionLeg(
                ticker=ticker,
                quantity=quantity,
                strike=strike,
                entry_spot=spot,
                entry_iv=implied,
                entry_premium=mark.value,
                entry_vega=state.dollar_vega,
            )
        )

    open_leg(
        INDEX_TICKER,
        index_spot,
        index_iv,
        config.target_gross_vega,
        index_sign,
        config.index_dividend_yield,
    )
    if not states:
        return None
    for ticker in available:
        open_leg(
            ticker,
            price_series[ticker][entry_date],
            volatility_series[ticker][entry_date],
            weight * config.target_gross_vega,
            constituent_sign,
            config.constituent_dividend_yield,
        )
    if len(states) < 2:
        return None

    total_vega_pnl = 0.0
    volatility_pnl = 0.0
    rebalances = 0
    entry_index_iv = index_iv
    entry_constituent_iv = float(
        np.average(
            [state.implied_volatility for state in states if state.ticker != INDEX_TICKER],
            weights=[abs(state.dollar_vega) for state in states if state.ticker != INDEX_TICKER],
        )
    )

    for step in range(1, config.holding_days + 1):
        day = dates[entry_index + step]
        remaining = config.horizon_years(config.holding_days - step)
        day_vega_changes: list[tuple[float, float]] = []

        for state in states:
            spot = price_series[state.ticker].get(day, state.spot)
            implied = volatility_series[state.ticker].get(day, state.implied_volatility)
            mark = straddle_mark(spot, state.strike, rate, state.dividend_yield, implied, remaining)

            change_in_value = (mark.value - state.mark.value) * state.quantity
            hedge_pnl = state.hedge_shares * (spot - state.spot)
            option_cash = state.quantity * state.mark.value
            financing = -rate * (option_cash + state.hedge_shares * state.spot) * DAY_FRACTION
            state.pnl += (change_in_value + hedge_pnl + financing) * costs.contract_multiplier

            change_in_iv = implied - state.implied_volatility
            day_vega_changes.append((state.dollar_vega, change_in_iv))
            total_vega_pnl += state.dollar_vega * change_in_iv

            if step < config.holding_days:
                target_shares = -state.quantity * mark.delta
                traded = abs(target_shares - state.hedge_shares)
                cost += (
                    traded * spot * costs.hedge_cost_bps(state.ticker) / BASIS_POINTS
                ) * costs.contract_multiplier
                state.hedge_shares = target_shares
                rebalances += 1

            state.spot, state.implied_volatility, state.mark = spot, implied, mark

        gross = sum(abs(vega) for vega, _ in day_vega_changes)
        if gross > 0:
            average_change = sum(abs(vega) * change for vega, change in day_vega_changes) / gross
            net_vega = sum(vega for vega, _ in day_vega_changes)
            volatility_pnl += net_vega * average_change

    exit_date = dates[exit_index]
    index_state = next(state for state in states if state.ticker == INDEX_TICKER)
    constituent_states = [state for state in states if state.ticker != INDEX_TICKER]
    index_pnl = index_state.pnl
    constituent_pnl = sum(state.pnl for state in constituent_states)
    gross_pnl = index_pnl + constituent_pnl
    net_pnl = gross_pnl - cost

    long_premium = sum(
        leg.quantity * leg.entry_premium * costs.contract_multiplier
        for leg in legs
        if leg.quantity > 0
    )
    short_notional = sum(
        abs(leg.quantity) * leg.entry_spot * costs.contract_multiplier
        for leg in legs
        if leg.quantity < 0
    )
    capital = long_premium + config.short_option_margin_fraction * short_notional

    exit_constituent_iv = float(
        np.average(
            [state.implied_volatility for state in constituent_states],
            weights=[abs(state.dollar_vega) or 1.0 for state in constituent_states],
        )
    )

    return DispersionTrade(
        entry_date=entry_date,
        exit_date=exit_date,
        direction=direction,
        entry_zscore=entry_zscore,
        entry_implied_correlation=entry_implied_correlation,
        realized_correlation=realized_correlation,
        correlation_change=(
            realized_correlation - entry_implied_correlation
            if realized_correlation is not None
            else None
        ),
        index_entry_iv=entry_index_iv,
        mean_constituent_entry_iv=entry_constituent_iv,
        index_iv_change=index_state.implied_volatility - entry_index_iv,
        mean_constituent_iv_change=exit_constituent_iv - entry_constituent_iv,
        legs=tuple(legs),
        gross_vega=config.target_gross_vega,
        net_pnl=net_pnl,
        correlation_pnl=total_vega_pnl - volatility_pnl,
        volatility_pnl=volatility_pnl,
        residual_pnl=gross_pnl - total_vega_pnl,
        index_pnl=index_pnl,
        constituent_pnl=constituent_pnl,
        cost=cost,
        capital=capital,
        return_on_capital=net_pnl / capital if capital > 0 else float("nan"),
        rebalances=rebalances,
        in_sample=entry_date <= config.train_cutoff(),
    )


def attribution_error(trade: DispersionTrade) -> float:
    """Return how far the attribution is from the trade's actual profit before costs.

    The three components are defined to sum to the total, so this should be zero to floating-point
    precision. It is reported so the identity is checked rather than assumed.
    """
    parts = trade.correlation_pnl + trade.volatility_pnl + trade.residual_pnl
    return float(abs(parts - (trade.net_pnl + trade.cost)))


def trade_row(trade: DispersionTrade) -> Row:
    """Flatten a trade into an export row."""
    return {
        "entry_date": trade.entry_date,
        "exit_date": trade.exit_date,
        "direction": trade.direction,
        "entry_zscore": trade.entry_zscore,
        "entry_implied_correlation": trade.entry_implied_correlation,
        "realized_correlation": trade.realized_correlation,
        "correlation_change": trade.correlation_change,
        "index_entry_iv": trade.index_entry_iv,
        "mean_constituent_entry_iv": trade.mean_constituent_entry_iv,
        "index_iv_change": trade.index_iv_change,
        "mean_constituent_iv_change": trade.mean_constituent_iv_change,
        "iv_change_spread": trade.index_iv_change - trade.mean_constituent_iv_change,
        "legs": len(trade.legs),
        "gross_vega": trade.gross_vega,
        "index_pnl": trade.index_pnl,
        "constituent_pnl": trade.constituent_pnl,
        "correlation_pnl": trade.correlation_pnl,
        "volatility_pnl": trade.volatility_pnl,
        "residual_pnl": trade.residual_pnl,
        "cost": trade.cost,
        "net_pnl": trade.net_pnl,
        "capital": trade.capital,
        "return_on_capital": trade.return_on_capital,
        "rebalances": trade.rebalances,
        "in_sample": trade.in_sample,
        "attribution_error": attribution_error(trade),
    }


def run_backtest(
    panel: Sequence[Row],
    dates: Sequence[date],
    price_series: Mapping[str, LevelByDate],
    volatility_series: Mapping[str, LevelByDate],
    constituent_tickers: Sequence[str],
    config: StudyConfig,
    costs: CostModel,
) -> list[DispersionTrade]:
    """Run the scheduled, non-overlapping backtest.

    A trade opens only when the signal fires and no trade is already open. Overlapping entries
    would share most of one price path, inflating the sample without adding information.

    Args:
        panel: Panel rows carrying a signal, ascending by date.
        dates: Trading calendar the panel is aligned to.
        price_series: Closing prices keyed by ticker.
        volatility_series: Implied-volatility series keyed by ticker.
        constituent_tickers: Names eligible to carry a constituent leg.
        config: Study configuration.
        costs: Transaction-cost model.

    Returns:
        Completed trades in entry order.
    """
    positions = {day: index for index, day in enumerate(dates)}
    trades: list[DispersionTrade] = []
    next_available: date | None = None

    for row in panel:
        if row["signal"] not in (SHORT_CORRELATION, LONG_CORRELATION):
            continue
        if row["spread_zscore"] is None:
            continue
        if next_available is not None and row["date"] < next_available:
            continue
        entry_index = positions.get(row["date"])
        if entry_index is None or entry_index < config.warmup_days:
            continue

        trade = simulate_trade(
            dates,
            entry_index,
            row["signal"],
            row["spread_zscore"],
            row["implied_correlation"],
            row["forward_realized_correlation"],
            price_series,
            volatility_series,
            constituent_tickers,
            config,
            costs,
        )
        if trade is None:
            continue
        trades.append(trade)
        next_available = trade.exit_date

    logger.info("Simulated %d dispersion trades", len(trades))
    return trades
