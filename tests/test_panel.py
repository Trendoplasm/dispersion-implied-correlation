"""The daily correlation panel and its alignment."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from dispersion.config import MIN_CONSTITUENTS, StudyConfig
from dispersion.models import Constituent, LevelByDate, Table
from dispersion.panel import BasketReturns, build_panel, panel_summary

from .conftest import (
    KNOWN_CORRELATION,
    SHORT_WINDOW_TOLERANCE,
    WINDOW_TOLERANCE,
    cboe_series,
    synthetic_basket,
    synthetic_prices,
    trading_dates,
)

IMPLIED = 0.45


@pytest.fixture
def calendar() -> list[date]:
    return trading_dates(date(2013, 1, 2), 200)


@pytest.fixture
def basket_ref() -> list[Constituent]:
    return synthetic_basket()


@pytest.fixture
def price_map(basket_ref: list[Constituent], calendar: list[date]) -> dict[str, LevelByDate]:
    return synthetic_prices(basket_ref, calendar)


@pytest.fixture
def cboe(calendar: list[date]) -> dict[str, LevelByDate]:
    return cboe_series(calendar, implied_correlation=IMPLIED)


@pytest.fixture
def returns(
    basket_ref: list[Constituent], price_map: dict[str, LevelByDate], calendar: list[date]
) -> BasketReturns:
    return BasketReturns(basket_ref, price_map, calendar[0], calendar[-1])


class TestBasketReturns:
    def test_uses_the_index_calendar(self, returns: BasketReturns, calendar: list[date]) -> None:
        assert returns.dates == calendar

    def test_returns_are_one_shorter_than_dates(self, returns: BasketReturns) -> None:
        assert returns.returns.shape[1] == len(returns.dates) - 1

    def test_a_window_recovers_the_planted_correlation(self, returns: BasketReturns) -> None:
        from dispersion.correlation import average_pairwise_correlation

        window = returns.window(100, 60)
        assert window is not None
        assert average_pairwise_correlation(window) == pytest.approx(
            KNOWN_CORRELATION, abs=WINDOW_TOLERANCE
        )

    def test_a_window_before_the_history_starts_is_refused(self, returns: BasketReturns) -> None:
        assert returns.window(10, 60) is None

    def test_a_window_past_the_history_end_is_refused(self, returns: BasketReturns) -> None:
        assert returns.window(len(returns.dates) + 10, 60) is None

    def test_incomplete_constituents_are_excluded_from_a_window(
        self, basket_ref: list[Constituent], price_map: dict[str, LevelByDate], calendar: list[date]
    ) -> None:
        # A name that did not trade for part of the window is dropped from it, so a correlation is
        # never assembled out of different names on different days.
        gapped = {ticker: dict(series) for ticker, series in price_map.items()}
        for day in calendar[80:100]:
            del gapped[basket_ref[0].ticker][day]
        returns = BasketReturns(basket_ref, gapped, calendar[0], calendar[-1])
        window = returns.window(100, 60)
        assert window is not None
        assert window.shape[0] == len(basket_ref) - 1

    def test_too_few_complete_constituents_refuses_the_window(
        self, basket_ref: list[Constituent], calendar: list[date]
    ) -> None:
        small = basket_ref[: MIN_CONSTITUENTS - 1]
        prices = synthetic_prices(small, calendar)
        returns = BasketReturns(small, prices, calendar[0], calendar[-1])
        assert returns.window(100, 60) is None

    def test_an_empty_calendar_is_rejected(
        self, basket_ref: list[Constituent], price_map: dict[str, LevelByDate]
    ) -> None:
        with pytest.raises(ValueError, match="No index prices"):
            BasketReturns(basket_ref, price_map, date(2013, 1, 2), date(2000, 1, 1))


class TestBuildPanel:
    @pytest.fixture
    def panel(
        self, returns: BasketReturns, cboe: dict[str, LevelByDate], config: StudyConfig
    ) -> Table:
        return build_panel(returns, cboe, config)

    def test_the_premium_is_exactly_implied_minus_realised(self, panel: Table) -> None:
        # This identity is what the panel code owns, so it is asserted exactly. How close the
        # measured correlation lands to the planted one is a property of the fixture's window
        # length, tested separately below.
        assert panel
        measured = [row for row in panel if row["correlation_premium"] is not None]
        assert measured
        for row in measured:
            assert row["implied_correlation"] == pytest.approx(IMPLIED)
            assert row["correlation_premium"] == pytest.approx(
                row["implied_correlation"] - row["forward_realized_correlation"]
            )

    def test_the_realised_legs_land_on_the_planted_correlation(self, panel: Table) -> None:
        measured = [row for row in panel if row["correlation_premium"] is not None]
        trailing = np.array([row["trailing_realized_correlation"] for row in measured])
        forward = np.array([row["forward_realized_correlation"] for row in measured])
        # The trailing window is 60 observations and the forward window 21, so the shorter one is
        # allowed to sit further from the planted value.
        assert float(trailing.mean()) == pytest.approx(KNOWN_CORRELATION, abs=WINDOW_TOLERANCE)
        assert float(forward.mean()) == pytest.approx(KNOWN_CORRELATION, abs=SHORT_WINDOW_TOLERANCE)

    def test_the_spread_uses_the_trailing_window(self, panel: Table) -> None:
        for row in panel:
            assert row["correlation_spread"] == pytest.approx(
                row["implied_correlation"] - row["trailing_realized_correlation"]
            )

    def test_the_identity_measure_agrees_on_a_uniform_basket(self, panel: Table) -> None:
        # Equal weights and equal volatilities make the two realised measures coincide, which is a
        # useful internal consistency check on the identity implementation.
        values = [
            row["identity_realized_correlation"]
            for row in panel
            if row["identity_realized_correlation"] is not None
        ]
        assert values
        assert float(np.mean(values)) == pytest.approx(KNOWN_CORRELATION, abs=1e-3)

    def test_carries_the_supporting_series(self, panel: Table) -> None:
        row = panel[0]
        assert row["implied_correlation_3m"] == pytest.approx(IMPLIED + 0.04)
        assert row["implied_correlation_6m"] == pytest.approx(IMPLIED + 0.07)
        assert row["dispersion_index"] == pytest.approx(1.0 - IMPLIED)
        assert row["index_iv"] == pytest.approx(0.18)

    def test_records_how_many_constituents_were_used(self, panel: Table) -> None:
        for row in panel:
            assert row["constituents_used"] >= MIN_CONSTITUENTS

    def test_skips_days_without_an_implied_quote(
        self, returns: BasketReturns, cboe: dict[str, LevelByDate], config: StudyConfig
    ) -> None:
        trimmed = dict(cboe)
        trimmed["COR1M"] = {}
        assert build_panel(returns, trimmed, config) == []

    def test_marks_the_in_sample_period(self, panel: Table, config: StudyConfig) -> None:
        for row in panel:
            assert row["in_sample"] == (row["date"] <= config.train_cutoff())


class TestPanelSummary:
    def test_summarises_a_known_panel(
        self, returns: BasketReturns, cboe: dict[str, LevelByDate], config: StudyConfig
    ) -> None:
        panel = build_panel(returns, cboe, config)
        summary = panel_summary(panel, "All days")
        assert summary["mean_implied_correlation"] == pytest.approx(IMPLIED)
        assert summary["mean_forward_realized_correlation"] == pytest.approx(
            KNOWN_CORRELATION, abs=SHORT_WINDOW_TOLERANCE
        )
        assert summary["mean_correlation_premium"] == pytest.approx(
            IMPLIED - KNOWN_CORRELATION, abs=SHORT_WINDOW_TOLERANCE
        )
        # Implied exceeds realised on every observation of this fixture.
        assert summary["pct_positive_premium"] == pytest.approx(1.0)

    def test_a_group_without_a_measurable_premium_is_refused(self) -> None:
        with pytest.raises(ValueError, match="No measurable premium"):
            panel_summary([{"correlation_premium": None}], "All days")
