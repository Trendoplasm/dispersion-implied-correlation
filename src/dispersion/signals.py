"""The trading signal, and the discipline that keeps it honest.

The signal is the one the original study specifies: the spread between implied correlation and
trailing realised correlation, standardised against its own recent history. A high z-score means
the option market is pricing more co-movement than the basket has been delivering, which is the
case for selling index volatility against constituent volatility.

Every input is strictly backward-looking. The standardising mean and standard deviation use a
window that ends the day *before* the observation, so the value being scored is never part of the
distribution it is scored against -- a subtle form of look-ahead that would otherwise shrink every
z-score toward zero and flatter the signal.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from dispersion.config import StudyConfig
from dispersion.models import Row, Table

logger = logging.getLogger(__name__)

#: Direction a trade takes when the signal fires.
SHORT_CORRELATION = "short_correlation"
LONG_CORRELATION = "long_correlation"
NO_TRADE = "no_trade"

#: Minimum observations in the standardising window before a z-score is reported.
MIN_ZSCORE_OBSERVATIONS = 60


def add_zscores(rows: Sequence[Row], config: StudyConfig) -> Table:
    """Attach a lagged z-score and a signal direction to each panel row.

    Args:
        rows: Panel rows, ascending by date.
        config: Window lengths and the z-score thresholds.

    Returns:
        The same rows with ``spread_zscore``, ``signal`` and the standardising statistics added.
        Rows before the window is populated carry a None z-score and ``no_trade``.
    """
    spreads = [row["correlation_spread"] for row in rows]
    enriched: Table = []

    for position, row in enumerate(rows):
        window_start = max(0, position - config.zscore_window_days)
        # Excludes `position` itself: the value being scored must not shape its own benchmark.
        history = np.array(spreads[window_start:position], dtype=float)

        mean = deviation = zscore = None
        signal = NO_TRADE
        if history.size >= MIN_ZSCORE_OBSERVATIONS:
            mean = float(np.mean(history))
            deviation = float(np.std(history, ddof=1))
            if deviation > 0:
                zscore = (row["correlation_spread"] - mean) / deviation
                if zscore > config.short_correlation_z:
                    signal = SHORT_CORRELATION
                elif zscore < config.long_correlation_z:
                    signal = LONG_CORRELATION

        enriched.append(
            {
                **row,
                "spread_mean_lagged": mean,
                "spread_sd_lagged": deviation,
                "spread_zscore": zscore,
                "signal": signal,
            }
        )

    fired = sum(entry["signal"] != NO_TRADE for entry in enriched)
    logger.info("Signal fired on %d of %d panel days", fired, len(enriched))
    return enriched


def signal_summary(rows: Sequence[Row]) -> Table:
    """Summarise the realised premium conditional on each signal state.

    Args:
        rows: Panel rows carrying a signal and a forward-looking premium.

    Returns:
        One row per signal state. This is the honest test of whether the signal identifies
        anything: if the premium after a short-correlation signal is no better than the
        unconditional average, the signal is decoration.
    """
    summary: Table = []
    for state in (SHORT_CORRELATION, NO_TRADE, LONG_CORRELATION, "All days"):
        if state == "All days":
            subset = [row for row in rows if row["correlation_premium"] is not None]
        else:
            subset = [
                row
                for row in rows
                if row["signal"] == state and row["correlation_premium"] is not None
            ]
        if not subset:
            summary.append({"signal": state, "n": 0})
            continue
        premium = np.array([row["correlation_premium"] for row in subset], dtype=float)
        summary.append(
            {
                "signal": state,
                "n": len(subset),
                "mean_premium": float(np.mean(premium)),
                "median_premium": float(np.median(premium)),
                "pct_positive_premium": float(np.mean(premium > 0)),
                "mean_implied_correlation": float(
                    np.mean([row["implied_correlation"] for row in subset])
                ),
                "mean_zscore": float(
                    np.mean(
                        [row["spread_zscore"] for row in subset if row["spread_zscore"] is not None]
                    )
                )
                if any(row["spread_zscore"] is not None for row in subset)
                else None,
                "mean_index_iv": float(
                    np.mean([row["index_iv"] for row in subset if row["index_iv"] is not None])
                )
                if any(row["index_iv"] is not None for row in subset)
                else None,
            }
        )
    return summary


def volatility_regime(index_iv: float, boundaries: tuple[float, float]) -> str:
    """Classify a day by the level of index implied volatility.

    Args:
        index_iv: Index implied volatility on the day.
        boundaries: Lower and upper cut points.

    Returns:
        The regime label.
    """
    lower, upper = boundaries
    if index_iv <= lower:
        return "Low volatility"
    if index_iv <= upper:
        return "Middle volatility"
    return "High volatility"


def regime_boundaries(rows: Sequence[Row]) -> tuple[float, float]:
    """Return the tercile cut points of index implied volatility over the panel.

    Raises:
        ValueError: If no row carries an index volatility level.
    """
    levels = [row["index_iv"] for row in rows if row.get("index_iv") is not None]
    if not levels:
        raise ValueError("No index implied-volatility levels to form regimes from")
    lower, upper = np.quantile(np.asarray(levels, dtype=float), [1 / 3, 2 / 3])
    return float(lower), float(upper)
