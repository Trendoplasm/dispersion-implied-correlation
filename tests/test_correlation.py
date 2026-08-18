"""Implied and realised correlation, checked against exactly known values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dispersion.config import MIN_WINDOW_OBSERVATIONS, TRADING_DAYS_PER_YEAR
from dispersion.correlation import (
    annualized_volatility,
    average_pairwise_correlation,
    dispersion_from_correlation,
    identity_implied_correlation,
    log_returns,
)

from .conftest import DAILY_SCALE, KNOWN_CORRELATION, correlated_returns


class TestLogReturns:
    def test_computes_log_differences(self) -> None:
        assert log_returns([100.0, 110.0]) == pytest.approx([math.log(1.1)])

    def test_returns_are_additive(self) -> None:
        prices = [100.0, 105.0, 99.0, 120.0]
        assert float(np.sum(log_returns(prices))) == pytest.approx(math.log(1.2))

    @pytest.mark.parametrize("prices", [[], [100.0]])
    def test_too_short_a_window_yields_nothing(self, prices: list[float]) -> None:
        assert log_returns(prices).size == 0

    @pytest.mark.parametrize("prices", [[100.0, 0.0], [100.0, -5.0]])
    def test_nonpositive_prices_are_rejected(self, prices: list[float]) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            log_returns(prices)


class TestAnnualizedVolatility:
    def test_uses_the_uncentred_second_moment(self) -> None:
        # Realised variance is the mean *squared* return, not the sample variance: over a short
        # window the estimated mean return is almost entirely noise, and subtracting it removes
        # signal rather than bias. The fixture scales rows by their sample standard deviation
        # (which divides by T-1), so the two differ by exactly sqrt((T-1)/T) -- a relationship
        # worth pinning down rather than absorbing into a loose tolerance.
        returns = correlated_returns(4, 0.0)[0]
        observations = returns.size
        expected = (
            DAILY_SCALE
            * math.sqrt(TRADING_DAYS_PER_YEAR)
            * math.sqrt((observations - 1) / observations)
        )
        assert annualized_volatility(returns) == pytest.approx(expected, rel=1e-9)

    def test_a_flat_series_has_no_volatility(self) -> None:
        assert annualized_volatility(np.zeros(60)) == pytest.approx(0.0)

    def test_too_short_a_window_is_reported_as_missing(self) -> None:
        assert annualized_volatility(np.zeros(MIN_WINDOW_OBSERVATIONS - 1)) is None


class TestAveragePairwiseCorrelation:
    @pytest.mark.parametrize("planted", [0.0, 0.15, 0.4, 0.75, 1.0])
    def test_recovers_the_planted_correlation_exactly(self, planted: float) -> None:
        # The Hadamard construction has no sampling error, so this is an equality, not a tendency.
        returns = correlated_returns(12, planted)
        assert average_pairwise_correlation(returns) == pytest.approx(planted, abs=1e-10)

    def test_identical_constituents_are_perfectly_correlated(self) -> None:
        single = correlated_returns(1, 0.0)
        stacked = np.repeat(single, 5, axis=0)
        assert average_pairwise_correlation(stacked) == pytest.approx(1.0)

    def test_opposite_constituents_are_perfectly_anticorrelated(self) -> None:
        single = correlated_returns(1, 0.0)
        pair = np.vstack([single, -single])
        assert average_pairwise_correlation(pair) == pytest.approx(-1.0)

    def test_motionless_constituents_are_dropped_not_treated_as_zero(self) -> None:
        # A constituent that never moved has no correlation with anything; counting it as zero
        # would drag the basket average toward zero for a reason that is not about the market.
        moving = correlated_returns(4, 1.0)
        with_flat = np.vstack([moving, np.zeros((2, moving.shape[1]))])
        assert average_pairwise_correlation(with_flat) == pytest.approx(1.0)

    def test_too_few_usable_constituents_is_reported_as_missing(self) -> None:
        flat = np.zeros((5, 60))
        assert average_pairwise_correlation(flat) is None

    def test_too_short_a_window_is_reported_as_missing(self) -> None:
        assert average_pairwise_correlation(np.ones((5, MIN_WINDOW_OBSERVATIONS - 1))) is None

    def test_a_one_dimensional_input_is_rejected(self) -> None:
        assert average_pairwise_correlation(np.zeros(60)) is None


class TestIdentityImpliedCorrelation:
    def test_two_equal_legs_have_a_closed_form(self) -> None:
        # With two equal weights and equal volatilities the identity reduces to
        #   rho = (sigma_basket^2 / sigma^2 - 1/2) * 2
        sigma, weight = 0.30, 0.5
        for planted in (0.0, 0.25, 0.5, 1.0):
            basket_variance = 2 * weight**2 * sigma**2 + 2 * weight**2 * sigma**2 * planted
            basket_volatility = math.sqrt(basket_variance)
            result = identity_implied_correlation(
                basket_volatility, [sigma, sigma], [weight, weight]
            )
            assert result == pytest.approx(planted)

    def test_perfectly_correlated_basket_returns_one(self) -> None:
        # If everything moves together, basket volatility is the weighted sum of volatilities.
        volatilities = [0.20, 0.30, 0.40]
        weights = [0.5, 0.3, 0.2]
        basket = sum(w * s for w, s in zip(weights, volatilities, strict=True))
        assert identity_implied_correlation(basket, volatilities, weights) == pytest.approx(1.0)

    def test_independent_basket_returns_zero(self) -> None:
        volatilities = [0.20, 0.30, 0.40]
        weights = [0.5, 0.3, 0.2]
        basket = math.sqrt(sum((w * s) ** 2 for w, s in zip(weights, volatilities, strict=True)))
        assert identity_implied_correlation(basket, volatilities, weights) == pytest.approx(0.0)

    def test_matches_the_average_pairwise_measure_on_a_uniform_basket(self) -> None:
        # When every constituent has the same weight and the same volatility the two measures
        # coincide. That they agree here, and differ elsewhere, is a property of the constructs.
        returns = correlated_returns(8, KNOWN_CORRELATION)
        volatilities = [annualized_volatility(row) or 0.0 for row in returns]
        weights = [1.0 / returns.shape[0]] * returns.shape[0]
        basket_volatility = annualized_volatility(returns.mean(axis=0))
        assert basket_volatility is not None
        identity = identity_implied_correlation(basket_volatility, volatilities, weights)
        assert identity == pytest.approx(KNOWN_CORRELATION, abs=1e-6)

    def test_single_constituent_has_no_correlation_to_solve_for(self) -> None:
        assert identity_implied_correlation(0.2, [0.2], [1.0]) is None

    def test_zero_volatility_everywhere_has_no_solution(self) -> None:
        assert identity_implied_correlation(0.0, [0.0, 0.0], [0.5, 0.5]) is None

    def test_ragged_inputs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            identity_implied_correlation(0.2, [0.2, 0.3], [1.0])

    def test_empty_inputs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one constituent"):
            identity_implied_correlation(0.2, [], [])

    @pytest.mark.parametrize(("basket", "volatilities"), [(-0.1, [0.2, 0.3]), (0.2, [-0.2, 0.3])])
    def test_negative_volatility_is_rejected(
        self, basket: float, volatilities: list[float]
    ) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            identity_implied_correlation(basket, volatilities, [0.5, 0.5])


def test_dispersion_is_the_complement_of_correlation() -> None:
    assert dispersion_from_correlation(0.3) == pytest.approx(0.7)
    assert dispersion_from_correlation(1.0) == pytest.approx(0.0)
