"""Summaries, tail statistics, and the resampled intervals."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from dispersion.aggregate import (
    attribution_table,
    bootstrap_interval,
    expected_shortfall,
    period_table,
    premium_by_regime,
    regime_table,
    strategy_summaries,
    summarize_trades,
    tail_tables,
)
from dispersion.config import TRADING_DAYS_PER_YEAR, StudyConfig
from dispersion.models import Table
from dispersion.signals import LONG_CORRELATION, SHORT_CORRELATION
from dispersion.trade import DispersionTrade


def trade(
    direction: str = SHORT_CORRELATION,
    *,
    net: float = 500.0,
    correlation: float = 900.0,
    volatility: float = 10.0,
    residual: float = 300.0,
    cost: float = 710.0,
    index_iv: float = 0.15,
    capital: float = 100_000.0,
    in_sample: bool = True,
) -> DispersionTrade:
    """Build a trade with directly specified outcomes."""
    return DispersionTrade(
        entry_date=date(2015, 1, 5),
        exit_date=date(2015, 2, 4),
        direction=direction,
        entry_zscore=1.2,
        entry_implied_correlation=0.45,
        realized_correlation=0.40,
        correlation_change=-0.05,
        index_entry_iv=index_iv,
        mean_constituent_entry_iv=0.28,
        index_iv_change=-0.01,
        mean_constituent_iv_change=0.0,
        legs=(),
        gross_vega=100_000.0,
        net_pnl=net,
        correlation_pnl=correlation,
        volatility_pnl=volatility,
        residual_pnl=residual,
        index_pnl=net + cost,
        constituent_pnl=0.0,
        cost=cost,
        capital=capital,
        return_on_capital=net / capital,
        rebalances=20,
        in_sample=in_sample,
    )


class TestExpectedShortfall:
    def test_averages_the_worst_tenth(self) -> None:
        assert expected_shortfall(list(range(100)), 0.10) == pytest.approx(4.5)

    def test_always_includes_at_least_one_observation(self) -> None:
        assert expected_shortfall([5.0, 9.0], 0.001) == pytest.approx(5.0)

    def test_is_no_greater_than_the_mean(self) -> None:
        values = [0.01, -0.13, 0.02, 0.03, -0.07]
        shortfall = expected_shortfall(values)
        assert shortfall is not None and shortfall <= float(np.mean(values))

    def test_empty_sample_reports_missing(self) -> None:
        assert expected_shortfall([]) is None


class TestSummarizeTrades:
    def test_reports_return_risk_and_tail(self) -> None:
        config = StudyConfig()
        trades = [trade(net=net) for net in (500.0, -3000.0, 800.0, 200.0)]
        summary = summarize_trades(trades, "All trades", config)
        assert summary["n"] == 4
        assert summary["win_rate"] == pytest.approx(0.75)
        assert summary["worst_trade_return"] == pytest.approx(-0.03)
        assert summary["total_net_pnl"] == pytest.approx(-1500.0)

    def test_annualises_from_the_holding_period(self) -> None:
        config = StudyConfig(holding_days=21)
        summary = summarize_trades([trade(net=1000.0)], "All trades", config)
        assert summary["annualized_mean_return"] == pytest.approx(0.01 * TRADING_DAYS_PER_YEAR / 21)

    def test_a_single_trade_has_no_dispersion(self) -> None:
        summary = summarize_trades([trade()], "All trades", StudyConfig())
        assert summary["sd_return_on_capital"] is None
        assert summary["sharpe_like_ratio"] is None

    def test_empty_group_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty trade group"):
            summarize_trades([], "All trades", StudyConfig())


class TestAttributionTable:
    def test_components_net_of_costs_sum_to_profit(self) -> None:
        # correlation + volatility + residual - cost == net, by construction.
        rows = attribution_table(
            [trade(net=500.0, correlation=900.0, volatility=10.0, residual=300.0, cost=710.0)]
        )
        assert rows
        for row in rows:
            assert row["check"] == pytest.approx(0.0, abs=1e-9)

    def test_covers_all_trades_and_each_direction(self) -> None:
        rows = attribution_table([trade(SHORT_CORRELATION), trade(LONG_CORRELATION)])
        assert [row["group"] for row in rows] == ["All trades", SHORT_CORRELATION, LONG_CORRELATION]

    def test_reports_how_small_the_volatility_term_is(self) -> None:
        # A vega-neutral structure should earn almost nothing from the average level of volatility;
        # this share is how the reader checks that claim.
        row = attribution_table([trade(correlation=900.0, volatility=10.0, residual=300.0)])[0]
        assert row["volatility_share_of_gross"] < 0.05


class TestRegimeAndPeriodTables:
    def test_regimes_partition_the_trades(self) -> None:
        trades = [trade(index_iv=level) for level in (0.10, 0.12, 0.18, 0.20, 0.30, 0.40)]
        rows = regime_table(trades, (0.15, 0.25), StudyConfig())
        assert sum(row["n"] for row in rows) == len(trades)

    def test_regimes_report_the_direction_mix(self) -> None:
        trades = [trade(SHORT_CORRELATION, index_iv=0.10), trade(LONG_CORRELATION, index_iv=0.10)]
        row = regime_table(trades, (0.15, 0.25), StudyConfig())[0]
        assert row["short_correlation_share"] == pytest.approx(0.5)

    def test_period_table_splits_in_and_out_of_sample(self) -> None:
        trades = [trade(in_sample=True), trade(in_sample=False), trade(in_sample=False)]
        rows = period_table(trades, StudyConfig())
        assert [row["n"] for row in rows] == [1, 2]

    def test_period_table_omits_an_empty_period(self) -> None:
        rows = period_table([trade(in_sample=True)], StudyConfig())
        assert len(rows) == 1


class TestTailTables:
    def test_lists_the_worst_and_best_trades(self) -> None:
        trades = [trade(net=net) for net in (-5000.0, 100.0, 900.0, -200.0)]
        worst, best = tail_tables(trades)
        assert worst[0]["net_pnl"] == pytest.approx(-5000.0)
        assert best[0]["net_pnl"] == pytest.approx(900.0)

    def test_each_table_runs_from_most_extreme(self) -> None:
        trades = [trade(net=net) for net in (-5000.0, 100.0, 900.0, -200.0)]
        worst, best = tail_tables(trades)
        assert [row["net_pnl"] for row in worst] == sorted(row["net_pnl"] for row in worst)
        assert [row["net_pnl"] for row in best] == sorted(
            (row["net_pnl"] for row in best), reverse=True
        )


class TestBootstrapInterval:
    def test_brackets_the_sample_mean(self) -> None:
        values = list(np.random.default_rng(0).normal(0.01, 0.05, 200))
        result = bootstrap_interval(values, np.random.default_rng(1), 500)
        assert result is not None
        low, high, mean = result
        assert low < mean < high

    def test_is_reproducible_under_a_fixed_seed(self) -> None:
        values = [0.01, -0.05, 0.02, 0.00, 0.03]
        first = bootstrap_interval(values, np.random.default_rng(9), 300)
        second = bootstrap_interval(values, np.random.default_rng(9), 300)
        assert first is not None
        assert first == second

    def test_a_constant_sample_has_a_degenerate_interval(self) -> None:
        result = bootstrap_interval([0.02] * 20, np.random.default_rng(2), 200)
        assert result is not None
        low, high, _ = result
        assert low == pytest.approx(0.02)
        assert high == pytest.approx(0.02)

    def test_an_empty_sample_has_no_interval(self) -> None:
        # There is no interval around the mean of nothing; None says so, a NaN would not.
        assert bootstrap_interval([], np.random.default_rng(3), 100) is None


class TestPremiumByRegime:
    def test_summarises_each_populated_regime(self) -> None:
        panel: Table = [
            {
                "index_iv": level,
                "implied_correlation": 0.40,
                "forward_realized_correlation": 0.35,
                "correlation_premium": 0.05,
            }
            for level in (0.10, 0.12, 0.18, 0.20, 0.30, 0.40)
        ]
        rows = premium_by_regime(panel, (0.15, 0.25))
        assert sum(row["n"] for row in rows) == len(panel)
        for row in rows:
            assert row["mean_premium"] == pytest.approx(0.05)
            assert row["pct_positive_premium"] == pytest.approx(1.0)

    def test_rows_without_a_premium_are_excluded(self) -> None:
        panel: Table = [
            {
                "index_iv": 0.10,
                "implied_correlation": 0.4,
                "forward_realized_correlation": 0.35,
                "correlation_premium": None,
            },
            {
                "index_iv": 0.10,
                "implied_correlation": 0.4,
                "forward_realized_correlation": 0.35,
                "correlation_premium": 0.05,
            },
        ]
        assert premium_by_regime(panel, (0.15, 0.25))[0]["n"] == 1


def test_strategy_summaries_cover_all_trades_and_both_directions() -> None:
    trades = [trade(SHORT_CORRELATION), trade(LONG_CORRELATION)]
    assert [row["group"] for row in strategy_summaries(trades, StudyConfig())] == [
        "All trades",
        SHORT_CORRELATION,
        LONG_CORRELATION,
    ]
