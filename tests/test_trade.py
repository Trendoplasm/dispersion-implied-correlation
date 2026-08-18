"""The dispersion structure: sizing, hedging, attribution, and the identity that ties them."""

from __future__ import annotations

from datetime import date

import pytest

from dispersion.config import INDEX_TICKER, CostModel, StudyConfig
from dispersion.models import LevelByDate
from dispersion.signals import LONG_CORRELATION, SHORT_CORRELATION
from dispersion.trade import DispersionTrade, attribution_error, simulate_trade, trade_row

from .conftest import (
    TRADEABLE,
    synthetic_basket,
    synthetic_prices,
    trading_dates,
)
from .conftest import (
    volatility_map as build_volatility_map,
)

ENTRY = 30


@pytest.fixture
def calendar() -> list[date]:
    return trading_dates(date(2013, 1, 2), 120)


@pytest.fixture
def price_map(calendar: list[date]) -> dict[str, LevelByDate]:
    return synthetic_prices(synthetic_basket(), calendar)


@pytest.fixture
def volatility_map(calendar: list[date]) -> dict[str, LevelByDate]:
    return build_volatility_map(calendar)


def run(
    calendar: list[date],
    price_map: dict[str, LevelByDate],
    volatility_map: dict[str, LevelByDate],
    config: StudyConfig,
    costs: CostModel,
    direction: str = SHORT_CORRELATION,
) -> DispersionTrade:
    trade = simulate_trade(
        calendar,
        ENTRY,
        direction,
        1.5,
        0.45,
        0.30,
        price_map,
        volatility_map,
        list(TRADEABLE),
        config,
        costs,
    )
    assert trade is not None
    return trade


class TestStructure:
    def test_short_correlation_sells_the_index_and_buys_constituents(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        trade = run(calendar, price_map, volatility_map, config, costs, SHORT_CORRELATION)
        index_leg = next(leg for leg in trade.legs if leg.ticker == INDEX_TICKER)
        assert index_leg.quantity < 0
        assert all(leg.quantity > 0 for leg in trade.legs if leg.ticker != INDEX_TICKER)

    def test_long_correlation_is_the_mirror_image(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        trade = run(calendar, price_map, volatility_map, config, costs, LONG_CORRELATION)
        index_leg = next(leg for leg in trade.legs if leg.ticker == INDEX_TICKER)
        assert index_leg.quantity > 0
        assert all(leg.quantity < 0 for leg in trade.legs if leg.ticker != INDEX_TICKER)

    def test_index_leg_is_sized_to_the_target_vega(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        trade = run(calendar, price_map, volatility_map, config, costs)
        index_leg = next(leg for leg in trade.legs if leg.ticker == INDEX_TICKER)
        assert abs(index_leg.entry_vega) == pytest.approx(config.target_gross_vega, rel=1e-9)

    def test_constituent_legs_together_offset_the_index_vega(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        # This is what makes it a correlation trade: a parallel shift in every implied volatility
        # has no first-order effect, so only the spread between the legs can pay.
        trade = run(calendar, price_map, volatility_map, config, costs)
        net_vega = sum(leg.entry_vega for leg in trade.legs)
        assert net_vega == pytest.approx(0.0, abs=1e-6)

    def test_every_tradeable_name_carries_a_leg(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        trade = run(calendar, price_map, volatility_map, config, costs)
        assert {leg.ticker for leg in trade.legs} == {INDEX_TICKER, *TRADEABLE}

    def test_strikes_sit_at_the_entry_forward(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        trade = run(calendar, price_map, volatility_map, config, costs)
        for leg in trade.legs:
            assert leg.strike == pytest.approx(leg.entry_spot, rel=0.01)

    def test_holds_for_the_configured_period(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        trade = run(calendar, price_map, volatility_map, config, costs)
        assert trade.entry_date == calendar[ENTRY]
        assert trade.exit_date == calendar[ENTRY + config.holding_days]


class TestAttribution:
    def test_components_sum_to_profit_before_costs(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        for direction in (SHORT_CORRELATION, LONG_CORRELATION):
            trade = run(calendar, price_map, volatility_map, config, costs, direction)
            assert attribution_error(trade) < 1e-6

    def test_constant_volatility_leaves_no_first_order_volatility_profit(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        free_costs: CostModel,
    ) -> None:
        # Every implied volatility is flat in the fixture, so neither the average level nor its
        # dispersion moves, and both volatility terms must vanish exactly.
        trade = run(calendar, price_map, volatility_map, config, free_costs)
        assert trade.correlation_pnl == pytest.approx(0.0)
        assert trade.volatility_pnl == pytest.approx(0.0)

    def test_a_relative_volatility_move_lands_in_correlation_not_volatility(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        free_costs: CostModel,
    ) -> None:
        # Drop index volatility only. That is a fall in implied correlation, and a short-correlation
        # position should book it under the correlation term.
        shifted = dict(volatility_map)
        index_series = dict(volatility_map[INDEX_TICKER])
        for day in list(index_series)[ENTRY + 5 :]:
            index_series[day] = 0.12
        shifted[INDEX_TICKER] = index_series
        trade = run(calendar, price_map, shifted, config, free_costs)
        assert trade.correlation_pnl > 0
        assert abs(trade.volatility_pnl) < abs(trade.correlation_pnl)
        assert attribution_error(trade) < 1e-6

    def test_a_parallel_volatility_move_is_not_booked_as_correlation(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        free_costs: CostModel,
    ) -> None:
        # Shift every leg's volatility by the same amount: a vega-neutral structure should be
        # indifferent, and the correlation term in particular should stay near zero.
        shifted = {}
        for ticker, series in volatility_map.items():
            moved = dict(series)
            for day in list(moved)[ENTRY + 5 :]:
                moved[day] = moved[day] + 0.05
            shifted[ticker] = moved
        trade = run(calendar, price_map, shifted, config, free_costs)
        assert abs(trade.correlation_pnl) < 0.02 * config.target_gross_vega
        assert attribution_error(trade) < 1e-6


class TestCostsAndCapital:
    def test_costs_reduce_profit(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
        free_costs: CostModel,
    ) -> None:
        charged = run(calendar, price_map, volatility_map, config, costs)
        free = run(calendar, price_map, volatility_map, config, free_costs)
        assert charged.cost > 0
        assert free.cost == 0
        assert charged.net_pnl < free.net_pnl

    def test_doubling_every_cost_roughly_doubles_the_bill(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        doubled = CostModel(
            option_commission=costs.option_commission * 2,
            option_half_spread=costs.option_half_spread * 2,
            index_hedge_cost_bps=costs.index_hedge_cost_bps * 2,
            stock_hedge_cost_bps=costs.stock_hedge_cost_bps * 2,
        )
        base = run(calendar, price_map, volatility_map, config, costs)
        stressed = run(calendar, price_map, volatility_map, config, doubled)
        assert stressed.cost == pytest.approx(2 * base.cost, rel=1e-9)

    def test_capital_is_long_premium_plus_a_share_of_short_notional(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        trade = run(calendar, price_map, volatility_map, config, costs)
        long_premium = sum(
            leg.quantity * leg.entry_premium * costs.contract_multiplier
            for leg in trade.legs
            if leg.quantity > 0
        )
        short_notional = sum(
            abs(leg.quantity) * leg.entry_spot * costs.contract_multiplier
            for leg in trade.legs
            if leg.quantity < 0
        )
        assert trade.capital == pytest.approx(
            long_premium + config.short_option_margin_fraction * short_notional
        )

    def test_return_is_profit_over_capital(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        trade = run(calendar, price_map, volatility_map, config, costs)
        assert trade.return_on_capital == pytest.approx(trade.net_pnl / trade.capital)

    def test_leg_profits_sum_to_the_gross(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        trade = run(calendar, price_map, volatility_map, config, costs)
        assert trade.index_pnl + trade.constituent_pnl == pytest.approx(trade.net_pnl + trade.cost)


class TestGuards:
    def test_returns_none_when_the_window_runs_past_the_data(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        assert (
            simulate_trade(
                calendar,
                len(calendar) - 5,
                SHORT_CORRELATION,
                1.0,
                0.4,
                0.3,
                price_map,
                volatility_map,
                list(TRADEABLE),
                config,
                costs,
            )
            is None
        )

    def test_returns_none_without_an_index_quote(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        stripped = dict(volatility_map)
        stripped[INDEX_TICKER] = {
            day: level
            for day, level in volatility_map[INDEX_TICKER].items()
            if day != calendar[ENTRY]
        }
        assert (
            simulate_trade(
                calendar,
                ENTRY,
                SHORT_CORRELATION,
                1.0,
                0.4,
                0.3,
                price_map,
                stripped,
                list(TRADEABLE),
                config,
                costs,
            )
            is None
        )

    def test_returns_none_without_any_constituent(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        assert (
            simulate_trade(
                calendar,
                ENTRY,
                SHORT_CORRELATION,
                1.0,
                0.4,
                0.3,
                price_map,
                volatility_map,
                [],
                config,
                costs,
            )
            is None
        )

    def test_a_missing_constituent_reduces_the_legs_but_still_trades(
        self,
        calendar: list[date],
        price_map: dict[str, LevelByDate],
        volatility_map: dict[str, LevelByDate],
        config: StudyConfig,
        costs: CostModel,
    ) -> None:
        # Losing one name should shrink the structure, not abandon the trade.
        trade = simulate_trade(
            calendar,
            ENTRY,
            SHORT_CORRELATION,
            1.0,
            0.4,
            0.3,
            price_map,
            volatility_map,
            list(TRADEABLE[:3]),
            config,
            costs,
        )
        assert trade is not None
        assert len(trade.legs) == 4


def test_trade_row_is_flat_and_self_consistent(
    calendar: list[date],
    price_map: dict[str, LevelByDate],
    volatility_map: dict[str, LevelByDate],
    config: StudyConfig,
    costs: CostModel,
) -> None:
    trade = run(calendar, price_map, volatility_map, config, costs)
    row = trade_row(trade)
    assert row["direction"] == SHORT_CORRELATION
    assert row["legs"] == len(trade.legs)
    error = row["attribution_error"]
    assert isinstance(error, float)
    assert error < 1e-6
    assert all(not isinstance(value, (list, dict, tuple)) for value in row.values())
