"""The daily correlation panel.

One row per trading day, carrying what the option market implied, what the basket went on to
deliver, and the supporting series. The forward-looking column is the one that defines the
premium, and by construction it cannot be used as a signal -- it is not knowable on the date it
sits against.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np

from dispersion.config import (
    DISPERSION_INDEX,
    INDEX_TICKER,
    INDEX_VOLATILITY_INDEX,
    MIN_CONSTITUENTS,
    StudyConfig,
)
from dispersion.correlation import (
    annualized_volatility,
    average_pairwise_correlation,
    identity_implied_correlation,
    log_returns,
)
from dispersion.models import Constituent, LevelByDate, Row, Table

logger = logging.getLogger(__name__)


class BasketReturns:
    """Aligned constituent returns on a single trading calendar.

    The index price series defines the calendar. A constituent missing on a date contributes no
    return there rather than a zero, because a zero would look like a day the stock did not move.

    Attributes:
        dates: Ascending trading dates.
        tickers: Constituents in matrix-row order.
        returns: Constituents by returns; ``nan`` where a return is unavailable.
        index_returns: Index log returns aligned to ``dates[1:]``.
        equal_weight_returns: Equal-weighted basket log returns aligned to ``dates[1:]``.
    """

    def __init__(
        self,
        constituents: Sequence[Constituent],
        price_series: Mapping[str, LevelByDate],
        start: date,
        end: date,
    ) -> None:
        """Align every constituent onto the index's trading calendar.

        Args:
            constituents: Basket members.
            price_series: Closing prices keyed by ticker, including the index.
            start: First date of the study period; history before it is kept for trailing windows.
            end: Last date of the study period.

        Raises:
            ValueError: If the calendar comes out empty.
        """
        index_prices = price_series[INDEX_TICKER]
        self.dates = sorted(day for day in index_prices if day <= end)
        if not self.dates:
            raise ValueError("No index prices within the study period")
        self.tickers = [item.ticker for item in constituents]

        levels = np.full((len(self.tickers), len(self.dates)), np.nan)
        for row, ticker in enumerate(self.tickers):
            series = price_series[ticker]
            for column, day in enumerate(self.dates):
                price = series.get(day)
                if price is not None:
                    levels[row, column] = price

        with np.errstate(invalid="ignore"):
            self.returns = np.diff(np.log(levels), axis=1)
        self.index_returns = log_returns([index_prices[day] for day in self.dates])
        # Equal-weighted basket return: the average of whatever constituents traded that day. A day
        # on which none of them traded is left as NaN explicitly, rather than letting numpy warn
        # about averaging an empty slice.
        available = np.isfinite(self.returns)
        counts = available.sum(axis=0)
        totals = np.where(available, self.returns, 0.0).sum(axis=0)
        self.equal_weight_returns = np.divide(
            totals, counts, out=np.full(counts.shape, np.nan), where=counts > 0
        )
        self.start_index = sum(day < start for day in self.dates)

    def window(self, end_index: int, length: int) -> np.ndarray | None:
        """Return the constituent returns in the ``length`` days ending at ``end_index``.

        Args:
            end_index: Position in :attr:`dates` of the window's last day.
            length: Number of returns to include.

        Returns:
            Rows for constituents with a complete window, or None if too few qualify. Requiring
            completeness within the window keeps a correlation estimate from being assembled out of
            different names on different days.
        """
        stop = end_index
        start = stop - length
        if start < 0 or stop > self.returns.shape[1]:
            return None
        block = self.returns[:, start:stop]
        complete = ~np.isnan(block).any(axis=1)
        if complete.sum() < MIN_CONSTITUENTS:
            return None
        return block[complete]


def build_panel(
    basket: BasketReturns,
    cboe_series: Mapping[str, LevelByDate],
    config: StudyConfig,
) -> Table:
    """Build the daily correlation panel.

    Args:
        basket: Aligned constituent returns.
        cboe_series: Loaded Cboe index histories.
        config: Study windows and period.

    Returns:
        One row per trading day inside the study period for which the implied leg and a trailing
        realised leg are both available.
    """
    implied = cboe_series["COR1M"]
    rows: Table = []

    for index, trading_date in enumerate(basket.dates):
        if index < basket.start_index:
            continue
        implied_correlation = implied.get(trading_date)
        if implied_correlation is None:
            continue

        # Returns are one shorter than dates, so the return ending on dates[index] is at index-1.
        return_position = index
        trailing = basket.window(return_position, config.realized_window_days)
        forward = basket.window(return_position + config.holding_days, config.holding_days)

        if trailing is None:
            continue
        trailing_correlation = average_pairwise_correlation(trailing)
        if trailing_correlation is None:
            continue
        forward_correlation = average_pairwise_correlation(forward) if forward is not None else None

        identity_realized = None
        weights = [1.0 / trailing.shape[0]] * trailing.shape[0]
        constituent_vols = [
            annualized_volatility(trailing[row]) or 0.0 for row in range(trailing.shape[0])
        ]
        basket_slice = basket.equal_weight_returns[
            return_position - config.realized_window_days : return_position
        ]
        basket_vol = annualized_volatility(basket_slice[~np.isnan(basket_slice)])
        if basket_vol is not None:
            identity_realized = identity_implied_correlation(basket_vol, constituent_vols, weights)

        rows.append(
            {
                "date": trading_date,
                "implied_correlation": implied_correlation,
                "implied_correlation_3m": cboe_series["COR3M"].get(trading_date),
                "implied_correlation_6m": cboe_series["COR6M"].get(trading_date),
                "dispersion_index": cboe_series[DISPERSION_INDEX].get(trading_date),
                "index_iv": cboe_series[INDEX_VOLATILITY_INDEX].get(trading_date),
                "trailing_realized_correlation": trailing_correlation,
                "forward_realized_correlation": forward_correlation,
                "identity_realized_correlation": identity_realized,
                "correlation_spread": implied_correlation - trailing_correlation,
                "correlation_premium": (
                    implied_correlation - forward_correlation
                    if forward_correlation is not None
                    else None
                ),
                "constituents_used": int(trailing.shape[0]),
                "in_sample": trading_date <= config.train_cutoff(),
            }
        )

    logger.info("Built correlation panel: %d observations", len(rows))
    return rows


def panel_summary(rows: Sequence[Row], label: str) -> Row:
    """Summarise a group of panel rows.

    Args:
        rows: Panel rows already filtered to the group.
        label: Value written to the ``group`` column.

    Returns:
        Means of both correlation legs, the premium and how often it was positive, and the
        observed term structure and dispersion levels.

    Raises:
        ValueError: If the group holds no row with a measurable premium.
    """
    with_premium = [row for row in rows if row["correlation_premium"] is not None]
    if not with_premium:
        raise ValueError(f"No measurable premium in group: {label}")

    premium = np.array([row["correlation_premium"] for row in with_premium], dtype=float)
    implied = np.array([row["implied_correlation"] for row in with_premium], dtype=float)
    realized = np.array([row["forward_realized_correlation"] for row in with_premium], dtype=float)

    def mean_of(key: str) -> float | None:
        values = [row[key] for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {
        "group": label,
        "n": len(with_premium),
        "first_date": min(row["date"] for row in with_premium),
        "last_date": max(row["date"] for row in with_premium),
        "mean_implied_correlation": float(np.mean(implied)),
        "mean_forward_realized_correlation": float(np.mean(realized)),
        "mean_correlation_premium": float(np.mean(premium)),
        "median_correlation_premium": float(np.median(premium)),
        "sd_correlation_premium": float(np.std(premium, ddof=1)) if premium.size > 1 else None,
        "pct_positive_premium": float(np.mean(premium > 0)),
        "mean_trailing_realized_correlation": mean_of("trailing_realized_correlation"),
        "mean_identity_realized_correlation": mean_of("identity_realized_correlation"),
        "mean_implied_correlation_3m": mean_of("implied_correlation_3m"),
        "mean_implied_correlation_6m": mean_of("implied_correlation_6m"),
        "mean_dispersion_index": mean_of("dispersion_index"),
        "mean_index_iv": mean_of("index_iv"),
    }
