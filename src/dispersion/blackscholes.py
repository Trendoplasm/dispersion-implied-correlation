"""Black-Scholes marks and Greeks for European options on a dividend-paying underlying.

The study needs these for two purposes: to mark a straddle daily so a delta-hedged return can be
accumulated, and to supply the Greeks that the profit decomposition attributes P&L to. Both uses
want the same convention, so it lives in one place.

Conventions
-----------
* Volatilities and rates are annual decimals (``0.20``, not ``20``).
* Maturities are year fractions.
* ``vega`` is per one unit of volatility, so a one-point move in a Cboe index (0.01 in decimal)
  changes value by ``0.01 * vega``.
* ``theta`` is per year; multiply by a day fraction for a daily figure.
* Every function accepts scalars or numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from scipy.special import ndtr

#: A scalar or an array of them.
Numeric: TypeAlias = float | np.ndarray

#: Below this maturity or volatility an option is treated as settled at intrinsic value. Both
#: quantities appear in denominators, and the limit is economically exact rather than a fudge.
EPSILON = 1e-12


def _normal_pdf(x: Numeric) -> np.ndarray:
    """Standard normal density."""
    values = np.asarray(x, dtype=float)
    return np.asarray(np.exp(-0.5 * values**2) / np.sqrt(2.0 * np.pi), dtype=float)


def _normal_cdf(x: Numeric) -> np.ndarray:
    """Standard normal cumulative distribution."""
    return np.asarray(ndtr(np.asarray(x, dtype=float)), dtype=float)


def forward_price(
    spot: Numeric, rate: float, dividend_yield: Numeric, maturity: Numeric
) -> Numeric:
    """Return the forward price of the underlying.

    Args:
        spot: Current underlying price.
        rate: Annual continuously compounded financing rate.
        dividend_yield: Annual continuous dividend yield.
        maturity: Year fraction to delivery.

    Returns:
        The forward price.
    """
    return np.asarray(
        np.asarray(spot, dtype=float)
        * np.exp(
            (rate - np.asarray(dividend_yield, dtype=float)) * np.asarray(maturity, dtype=float)
        ),
        dtype=float,
    )


def _d1_d2(
    spot: Numeric,
    strike: Numeric,
    rate: float,
    dividend_yield: Numeric,
    volatility: Numeric,
    maturity: Numeric,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``d1``, ``d2``, and the total volatility ``sigma * sqrt(T)``."""
    forward = np.asarray(forward_price(spot, rate, dividend_yield, maturity), dtype=float)
    strike = np.asarray(strike, dtype=float)
    total_volatility = np.maximum(
        np.asarray(volatility, dtype=float) * np.sqrt(np.asarray(maturity, dtype=float)),
        EPSILON,
    )
    d1 = np.log(forward / strike) / total_volatility + 0.5 * total_volatility
    return d1, d1 - total_volatility, total_volatility


@dataclass(frozen=True)
class StraddleMark:
    """A long straddle marked at one point in time.

    A straddle is one call plus one put at the same strike. The study trades it because it is the
    cleanest available exposure to volatility: at the money the two deltas nearly cancel, so what
    remains after hedging is dominated by variance rather than by direction.

    Attributes:
        value: Present value of the package, per share of underlying.
        delta: Sensitivity to the underlying price.
        gamma: Sensitivity of delta to the underlying price.
        vega: Sensitivity to one unit of volatility.
        theta: Sensitivity to the passage of one year.
        call_value: Present value of the call leg.
        put_value: Present value of the put leg.
    """

    value: float
    delta: float
    gamma: float
    vega: float
    theta: float
    call_value: float
    put_value: float


def straddle_mark(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
) -> StraddleMark:
    """Mark a long straddle and return its value together with its Greeks.

    At or below zero maturity, or at non-positive volatility, the package settles at intrinsic
    value and every Greek except delta is zero: there is no remaining optionality to be sensitive
    to.

    Args:
        spot: Current underlying price.
        strike: Common strike of the two legs.
        rate: Annual continuously compounded financing rate.
        dividend_yield: Annual continuous dividend yield.
        volatility: Annual implied volatility as a decimal.
        maturity: Year fraction to expiry.

    Returns:
        The marked package.
    """
    if maturity <= EPSILON or volatility <= EPSILON:
        call_intrinsic = max(spot - strike, 0.0)
        put_intrinsic = max(strike - spot, 0.0)
        return StraddleMark(
            value=call_intrinsic + put_intrinsic,
            delta=(1.0 if spot > strike else -1.0 if spot < strike else 0.0),
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            call_value=call_intrinsic,
            put_value=put_intrinsic,
        )

    d1, d2, total_volatility = _d1_d2(spot, strike, rate, dividend_yield, volatility, maturity)
    discount = float(np.exp(-rate * maturity))
    carry = float(np.exp(-dividend_yield * maturity))
    forward = float(forward_price(spot, rate, dividend_yield, maturity))
    cdf_d1, cdf_d2 = float(_normal_cdf(d1)), float(_normal_cdf(d2))
    pdf_d1 = float(_normal_pdf(d1))

    call_value = discount * (forward * cdf_d1 - strike * cdf_d2)
    put_value = discount * (strike * (1.0 - cdf_d2) - forward * (1.0 - cdf_d1))

    # The two legs share gamma and vega; their deltas partly offset.
    shared_gamma = carry * pdf_d1 / (spot * float(total_volatility))
    shared_vega = spot * carry * pdf_d1 * float(np.sqrt(maturity))

    call_delta = carry * cdf_d1
    put_delta = -carry * (1.0 - cdf_d1)

    time_decay = -spot * carry * pdf_d1 * volatility / (2.0 * float(np.sqrt(maturity)))
    call_theta = (
        time_decay - rate * strike * discount * cdf_d2 + dividend_yield * spot * carry * cdf_d1
    )
    put_theta = (
        time_decay
        + rate * strike * discount * (1.0 - cdf_d2)
        - dividend_yield * spot * carry * (1.0 - cdf_d1)
    )

    return StraddleMark(
        value=call_value + put_value,
        delta=call_delta + put_delta,
        gamma=2.0 * shared_gamma,
        vega=2.0 * shared_vega,
        theta=call_theta + put_theta,
        call_value=call_value,
        put_value=put_value,
    )


def straddle_value(
    spot: Numeric,
    strike: Numeric,
    rate: float,
    dividend_yield: Numeric,
    volatility: Numeric,
    maturity: Numeric,
) -> Numeric:
    """Return straddle present value, vectorised over any argument.

    Args:
        spot: Underlying price.
        strike: Common strike.
        rate: Annual continuously compounded financing rate.
        dividend_yield: Annual continuous dividend yield.
        volatility: Annual implied volatility as a decimal.
        maturity: Year fraction to expiry.

    Returns:
        Present value per share of underlying.
    """
    d1, d2, _ = _d1_d2(spot, strike, rate, dividend_yield, volatility, maturity)
    discount = np.exp(-rate * np.asarray(maturity, dtype=float))
    forward = np.asarray(forward_price(spot, rate, dividend_yield, maturity), dtype=float)
    strike = np.asarray(strike, dtype=float)
    cdf_d1, cdf_d2 = _normal_cdf(d1), _normal_cdf(d2)
    call = discount * (forward * cdf_d1 - strike * cdf_d2)
    put = discount * (strike * (1.0 - cdf_d2) - forward * (1.0 - cdf_d1))
    return np.asarray(call + put, dtype=float)
