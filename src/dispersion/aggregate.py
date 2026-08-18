"""Summaries of the backtest: overall, by direction, by regime, and in the tails.

A mean alone would misdescribe this strategy. It trades infrequently, its two directions behave
differently, and its losses cluster in the same episodes that make correlation spike. So every
summary pairs an average with a tail statistic, and the tail tables list individual trades rather
than only their average.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from dispersion.config import (
    BOOTSTRAP_LOWER_QUANTILE,
    BOOTSTRAP_UPPER_QUANTILE,
    EXTREME_TRADE_COUNT,
    REGIME_ORDER,
    TRADING_DAYS_PER_YEAR,
    StudyConfig,
)
from dispersion.models import Row, Table
from dispersion.signals import LONG_CORRELATION, SHORT_CORRELATION, volatility_regime
from dispersion.stats_utils import Samples, mean_or_none, median_or_none, std_dev_or_none
from dispersion.trade import DispersionTrade, trade_row

logger = logging.getLogger(__name__)

#: Fraction of the worst outcomes averaged to form the expected-shortfall statistic.
TAIL_FRACTION = 0.10

#: Groups reported in the strategy summary, in order.
DIRECTION_GROUPS: tuple[str, ...] = ("All trades", SHORT_CORRELATION, LONG_CORRELATION)


def expected_shortfall(values: Samples, fraction: float = TAIL_FRACTION) -> float | None:
    """Return the mean of the worst ``fraction`` of outcomes, or None for an empty sample."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    ordered = np.sort(array)
    count = max(1, int(np.ceil(fraction * ordered.size)))
    return float(np.mean(ordered[:count]))


def _select(trades: Sequence[DispersionTrade], group: str) -> list[DispersionTrade]:
    """Return the trades belonging to one reporting group."""
    if group == "All trades":
        return list(trades)
    return [trade for trade in trades if trade.direction == group]


def summarize_trades(trades: Sequence[DispersionTrade], label: str, config: StudyConfig) -> Row:
    """Summarise a group of trades.

    Args:
        trades: Trades in the group.
        label: Value written to the ``group`` column.
        config: Holding period, used to annualise.

    Returns:
        Return, risk, and attribution statistics.

    Raises:
        ValueError: If the group is empty.
    """
    if not trades:
        raise ValueError(f"Cannot summarise an empty trade group: {label}")

    returns = np.array([trade.return_on_capital for trade in trades], dtype=float)
    net = np.array([trade.net_pnl for trade in trades], dtype=float)
    periods_per_year = TRADING_DAYS_PER_YEAR / config.holding_days

    return {
        "group": label,
        "n": len(trades),
        "first_entry": min(trade.entry_date for trade in trades),
        "last_exit": max(trade.exit_date for trade in trades),
        "mean_net_pnl": float(np.mean(net)),
        "median_net_pnl": float(np.median(net)),
        "total_net_pnl": float(np.sum(net)),
        "mean_return_on_capital": float(np.mean(returns)),
        "median_return_on_capital": float(np.median(returns)),
        "sd_return_on_capital": std_dev_or_none(returns),
        "win_rate": float(np.mean(net > 0)),
        # Annualised from non-overlapping periods, so the scaling reflects actual frequency.
        "annualized_mean_return": float(np.mean(returns) * periods_per_year),
        "sharpe_like_ratio": (
            float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(periods_per_year))
            if returns.size > 1 and np.std(returns, ddof=1) > 0
            else None
        ),
        "worst_trade_return": float(np.min(returns)),
        "best_trade_return": float(np.max(returns)),
        "expected_shortfall_return": expected_shortfall(returns),
        "mean_entry_zscore": mean_or_none([trade.entry_zscore for trade in trades]),
        "mean_entry_implied_correlation": mean_or_none(
            [trade.entry_implied_correlation for trade in trades]
        ),
        "mean_correlation_change": mean_or_none(
            [t.correlation_change for t in trades if t.correlation_change is not None]
        ),
        "mean_cost": float(np.mean([trade.cost for trade in trades])),
        "mean_capital": float(np.mean([trade.capital for trade in trades])),
        "cost_share_of_capital": float(
            np.mean([trade.cost / trade.capital for trade in trades if trade.capital > 0])
        ),
    }


def strategy_summaries(trades: Sequence[DispersionTrade], config: StudyConfig) -> Table:
    """Summarise all trades and each direction."""
    summaries: Table = []
    for group in DIRECTION_GROUPS:
        subset = _select(trades, group)
        if subset:
            summaries.append(summarize_trades(subset, group, config))
    return summaries


def attribution_table(trades: Sequence[DispersionTrade]) -> Table:
    """Decompose average profit into correlation, volatility, residual, and cost.

    The four columns sum to net profit by construction; ``check`` reports the floating-point
    residual of that identity so the reader can verify it rather than trust it.
    """
    rows: Table = []
    for group in DIRECTION_GROUPS:
        subset = _select(trades, group)
        if not subset:
            continue
        correlation = float(np.mean([trade.correlation_pnl for trade in subset]))
        volatility = float(np.mean([trade.volatility_pnl for trade in subset]))
        residual = float(np.mean([trade.residual_pnl for trade in subset]))
        cost = float(np.mean([trade.cost for trade in subset]))
        net = float(np.mean([trade.net_pnl for trade in subset]))
        rows.append(
            {
                "group": group,
                "n": len(subset),
                "mean_correlation_pnl": correlation,
                "mean_volatility_pnl": volatility,
                "mean_residual_pnl": residual,
                "mean_cost": cost,
                "mean_net_pnl": net,
                "check": correlation + volatility + residual - cost - net,
                "mean_index_leg_pnl": float(np.mean([t.index_pnl for t in subset])),
                "mean_constituent_leg_pnl": float(np.mean([t.constituent_pnl for t in subset])),
                # Near zero is the point: a vega-neutral structure should not be paid for the
                # average level of volatility, only for the spread between its legs.
                "volatility_share_of_gross": (
                    abs(volatility) / (abs(correlation) + abs(volatility) + abs(residual))
                    if (abs(correlation) + abs(volatility) + abs(residual)) > 0
                    else None
                ),
            }
        )
    return rows


def regime_table(
    trades: Sequence[DispersionTrade], boundaries: tuple[float, float], config: StudyConfig
) -> Table:
    """Split trades by the level of index implied volatility at entry."""
    rows: Table = []
    for label in REGIME_ORDER:
        subset = [
            trade
            for trade in trades
            if volatility_regime(trade.index_entry_iv, boundaries) == label
        ]
        if not subset:
            logger.warning("Regime %s holds no trades", label)
            continue
        summary = summarize_trades(subset, label, config)
        summary["mean_index_entry_iv"] = float(np.mean([trade.index_entry_iv for trade in subset]))
        summary["short_correlation_share"] = float(
            np.mean([trade.direction == SHORT_CORRELATION for trade in subset])
        )
        rows.append(summary)
    return rows


def period_table(trades: Sequence[DispersionTrade], config: StudyConfig) -> Table:
    """Compare the threshold-selection period against the period that followed it.

    The thresholds come from the original study's specification rather than being fitted here, but
    reporting the split still matters: it shows whether the result depends on the years the rule
    was written against.
    """
    rows: Table = []
    for label, in_sample in (
        (f"In sample (to {config.train_end})", True),
        ("Out of sample (after)", False),
    ):
        subset = [trade for trade in trades if trade.in_sample is in_sample]
        if not subset:
            continue
        summary = summarize_trades(subset, label, config)
        rows.append(summary)
    return rows


def tail_tables(trades: Sequence[DispersionTrade]) -> tuple[Table, Table]:
    """Return the worst and best individual trades."""
    ordered = sorted(trades, key=lambda trade: trade.return_on_capital)
    worst = [trade_row(trade) for trade in ordered[:EXTREME_TRADE_COUNT]]
    best = [trade_row(trade) for trade in reversed(ordered[-EXTREME_TRADE_COUNT:])]
    return worst, best


def bootstrap_interval(
    values: Samples, rng: np.random.Generator, iterations: int
) -> tuple[float, float, float] | None:
    """Bootstrap a mean and its 95% interval.

    Args:
        values: Observations to resample.
        rng: Seeded generator, so a run is reproducible.
        iterations: Number of resamples.

    Returns:
        The lower bound, upper bound, and bootstrap mean, or None for an empty sample. There is no
        interval around the mean of nothing, and returning None says so rather than emitting a NaN
        that reads like a number.
    """
    sample = np.asarray(values, dtype=float)
    if sample.size == 0:
        return None
    draws = rng.integers(0, sample.size, size=(iterations, sample.size))
    means = sample[draws].mean(axis=1)
    return (
        float(np.quantile(means, BOOTSTRAP_LOWER_QUANTILE)),
        float(np.quantile(means, BOOTSTRAP_UPPER_QUANTILE)),
        float(np.mean(means)),
    )


def premium_by_regime(panel: Sequence[Row], boundaries: tuple[float, float]) -> Table:
    """Summarise the correlation premium within each volatility regime.

    This is the panel-level counterpart to :func:`regime_table`: it asks whether the premium itself
    depends on the regime, independently of whether the strategy traded.
    """
    rows: Table = []
    for label in REGIME_ORDER:
        subset = [
            row
            for row in panel
            if row.get("index_iv") is not None
            and row["correlation_premium"] is not None
            and volatility_regime(row["index_iv"], boundaries) == label
        ]
        if not subset:
            continue
        premium = np.array([row["correlation_premium"] for row in subset], dtype=float)
        rows.append(
            {
                "regime": label,
                "n": len(subset),
                "mean_index_iv": float(np.mean([row["index_iv"] for row in subset])),
                "mean_implied_correlation": float(
                    np.mean([row["implied_correlation"] for row in subset])
                ),
                "mean_forward_realized_correlation": float(
                    np.mean([row["forward_realized_correlation"] for row in subset])
                ),
                "mean_premium": float(np.mean(premium)),
                "median_premium": median_or_none(premium),
                "pct_positive_premium": float(np.mean(premium > 0)),
            }
        )
    return rows
