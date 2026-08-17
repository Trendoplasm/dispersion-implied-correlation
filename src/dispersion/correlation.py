"""Implied and realised correlation.

Two measures matter, and they are not the same thing.

**Implied correlation** is observed directly: Cboe publishes the average correlation the option
market prices between S&P 500 constituents. Nothing here has to infer it.

**Realised correlation** is computed from returns, and there are two defensible ways to do it:

*Average pairwise correlation* averages the correlation of every pair of constituents. It needs no
index weights, which is exactly why it is used here -- historical S&P 500 weights are not available
free of charge, and inventing them would put a guess at the centre of the measurement.

*Identity-implied correlation* inverts the variance identity that defines implied correlation,

    sigma_basket^2 = sum_i sum_j w_i w_j sigma_i sigma_j rho_ij

solving for the single ``rho`` consistent with an observed basket volatility. This is the
construct Cboe's index uses, so it is reported alongside as the like-for-like comparison.

The two differ systematically: the identity weights each pair by the product of its weights and
volatilities, so volatile, heavily weighted names dominate it. The study reports both rather than
choosing, because the gap between them is a real property of the basket and not an error.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from dispersion.config import MIN_WINDOW_OBSERVATIONS, TRADING_DAYS_PER_YEAR
from dispersion.stats_utils import Samples

logger = logging.getLogger(__name__)


def log_returns(prices: Sequence[float]) -> np.ndarray:
    """Return close-to-close log returns.

    Raises:
        ValueError: If any price is not positive, which makes a log return undefined.
    """
    values = np.asarray(prices, dtype=float)
    if values.size < 2:
        return np.empty(0, dtype=float)
    if np.any(values <= 0):
        raise ValueError("Log returns require strictly positive prices")
    return np.diff(np.log(values))


def annualized_volatility(returns: np.ndarray) -> float | None:
    """Return annualised volatility from a window of returns, or None if too short."""
    if returns.size < MIN_WINDOW_OBSERVATIONS:
        return None
    return float(np.sqrt(TRADING_DAYS_PER_YEAR * np.mean(returns**2)))


def average_pairwise_correlation(returns: np.ndarray) -> float | None:
    """Return the mean off-diagonal correlation of a return matrix.

    Args:
        returns: Constituents by observations.

    Returns:
        The mean of every distinct pairwise correlation, or None when the window is too short or
        holds fewer than two usable constituents. Constituents that never moved in the window are
        dropped, because their correlation with anything is undefined rather than zero.
    """
    if returns.ndim != 2 or returns.shape[1] < MIN_WINDOW_OBSERVATIONS:
        return None

    deviations = returns - returns.mean(axis=1, keepdims=True)
    scales = returns.std(axis=1, ddof=1)
    usable = scales > 0
    if usable.sum() < 2:
        return None

    standardized = deviations[usable] / scales[usable, None]
    count = int(usable.sum())
    matrix = (standardized @ standardized.T) / (returns.shape[1] - 1)
    off_diagonal_sum = float(matrix.sum() - np.trace(matrix))
    return off_diagonal_sum / (count * (count - 1))


def identity_implied_correlation(
    basket_volatility: float, volatilities: Samples, weights: Samples
) -> float | None:
    """Invert the variance identity for the single correlation consistent with a basket volatility.

    This is the construct Cboe's implied-correlation index uses. Applied to implied volatilities it
    yields implied correlation; applied to realised volatilities it yields realised correlation on
    the same footing.

    Args:
        basket_volatility: Volatility of the basket as a whole.
        volatilities: Constituent volatilities.
        weights: Constituent weights; they need not sum to one but must not be all zero.

    Returns:
        The implied correlation, or None when the cross term vanishes and the identity cannot be
        inverted -- with a single constituent, or with no volatility anywhere, there is no
        correlation to solve for.

    Raises:
        ValueError: If the inputs are ragged or contain a negative volatility.
    """
    sigma = np.asarray(volatilities, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if sigma.shape != weight.shape:
        raise ValueError("Volatilities and weights must have the same length")
    if sigma.size == 0:
        raise ValueError("Need at least one constituent")
    if np.any(sigma < 0) or basket_volatility < 0:
        raise ValueError("Volatilities cannot be negative")

    contributions = weight * sigma
    own_variance = float(np.sum(contributions**2))
    total_squared = float(np.sum(contributions) ** 2)
    # The cross term is the sum over distinct pairs, twice: (sum x)^2 - sum x^2.
    cross_term = total_squared - own_variance
    if cross_term <= 0:
        return None
    return (basket_volatility**2 - own_variance) / cross_term


def dispersion_from_correlation(correlation: float) -> float:
    """Return a dispersion score from a correlation level.

    Dispersion and correlation are two views of the same thing: when constituents move together
    there is little dispersion between them. Reporting ``1 - rho`` makes comparisons against
    Cboe's dispersion index read in the same direction.
    """
    return 1.0 - correlation
