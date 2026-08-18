"""The trading signal, and the guarantee that it cannot see its own future.

The property this module exists to protect is negative: the standardising window must end the day
*before* the observation it scores. Including the observation in its own benchmark is a subtle
look-ahead that shrinks every z-score toward zero and makes the signal look better behaved than it
is.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest

from dispersion.config import StudyConfig
from dispersion.signals import (
    LONG_CORRELATION,
    MIN_ZSCORE_OBSERVATIONS,
    NO_TRADE,
    SHORT_CORRELATION,
    add_zscores,
    regime_boundaries,
    signal_summary,
    volatility_regime,
)


def panel_rows(
    spreads: Sequence[float], premiums: Sequence[float] | None = None
) -> list[dict[str, Any]]:
    """Build minimal panel rows carrying a spread and optionally a realised premium."""
    start = date(2013, 1, 2)
    return [
        {
            "date": start + timedelta(days=index),
            "correlation_spread": spread,
            "implied_correlation": 0.35,
            "index_iv": 0.18,
            "correlation_premium": None if premiums is None else premiums[index],
            "forward_realized_correlation": 0.30,
            "in_sample": True,
        }
        for index, spread in enumerate(spreads)
    ]


class TestZscoreIsLagged:
    def test_the_scored_value_is_excluded_from_its_own_benchmark(self) -> None:
        # A long calm stretch then one large jump. If the jump were included in its own window it
        # would inflate the standard deviation and shrink its own z-score.
        spreads = [0.0] * 200 + [1.0]
        rows = add_zscores(panel_rows(spreads), StudyConfig(zscore_window_days=200))
        # The calm history has zero dispersion, so no z-score is defined for the jump.
        assert rows[-1]["spread_sd_lagged"] == pytest.approx(0.0)
        assert rows[-1]["spread_zscore"] is None

    def test_statistics_use_only_prior_observations(self) -> None:
        spreads = [float(index) for index in range(150)]
        config = StudyConfig(zscore_window_days=1000)
        rows = add_zscores(panel_rows(spreads), config)
        position = 120
        history = np.array(spreads[:position], dtype=float)
        assert rows[position]["spread_mean_lagged"] == pytest.approx(float(np.mean(history)))
        assert rows[position]["spread_sd_lagged"] == pytest.approx(float(np.std(history, ddof=1)))

    def test_future_observations_cannot_change_a_zscore(self) -> None:
        # The decisive test: append wildly different later data and every earlier z-score must be
        # bit-for-bit unchanged.
        spreads = list(np.linspace(-0.1, 0.1, 200))
        config = StudyConfig(zscore_window_days=120)
        before = add_zscores(panel_rows(spreads), config)
        after = add_zscores(panel_rows([*spreads, 50.0, -50.0, 99.0]), config)
        for original, extended in zip(before, after[: len(before)], strict=True):
            assert original["spread_zscore"] == extended["spread_zscore"]
            assert original["signal"] == extended["signal"]

    def test_no_zscore_before_the_window_is_populated(self) -> None:
        rows = add_zscores(panel_rows([0.01 * index for index in range(40)]), StudyConfig())
        assert all(row["spread_zscore"] is None for row in rows[:MIN_ZSCORE_OBSERVATIONS])
        assert all(row["signal"] == NO_TRADE for row in rows[:MIN_ZSCORE_OBSERVATIONS])


class TestSignalThresholds:
    @pytest.fixture
    def rows(self) -> list[dict[str, Any]]:
        rng = np.random.default_rng(0)
        spreads = list(rng.normal(0.0, 0.02, 300))
        return add_zscores(panel_rows(spreads), StudyConfig(zscore_window_days=150))

    def test_short_signal_needs_a_high_zscore(self, rows: list[dict[str, Any]]) -> None:
        config = StudyConfig()
        for row in rows:
            if row["signal"] == SHORT_CORRELATION:
                assert row["spread_zscore"] > config.short_correlation_z

    def test_long_signal_needs_a_low_zscore(self, rows: list[dict[str, Any]]) -> None:
        config = StudyConfig()
        for row in rows:
            if row["signal"] == LONG_CORRELATION:
                assert row["spread_zscore"] < config.long_correlation_z

    def test_the_middle_ground_does_not_trade(self, rows: list[dict[str, Any]]) -> None:
        config = StudyConfig()
        for row in rows:
            if row["spread_zscore"] is None:
                continue
            inside = config.long_correlation_z <= row["spread_zscore"] <= config.short_correlation_z
            assert (row["signal"] == NO_TRADE) == inside

    def test_both_signals_can_fire_over_a_long_sample(self, rows: list[dict[str, Any]]) -> None:
        states = {row["signal"] for row in rows}
        assert SHORT_CORRELATION in states
        assert LONG_CORRELATION in states

    def test_tighter_thresholds_fire_less_often(self) -> None:
        rng = np.random.default_rng(1)
        spreads = list(rng.normal(0.0, 0.02, 400))
        loose = add_zscores(
            panel_rows(spreads),
            StudyConfig(zscore_window_days=150, short_correlation_z=0.5, long_correlation_z=-0.5),
        )
        tight = add_zscores(
            panel_rows(spreads),
            StudyConfig(zscore_window_days=150, short_correlation_z=2.0, long_correlation_z=-2.0),
        )
        assert sum(r["signal"] != NO_TRADE for r in tight) < sum(
            r["signal"] != NO_TRADE for r in loose
        )


class TestSignalSummary:
    def test_reports_every_state_and_the_unconditional_row(self) -> None:
        rng = np.random.default_rng(2)
        spreads = list(rng.normal(0.0, 0.02, 300))
        premiums = list(rng.normal(0.0, 0.05, 300))
        rows = add_zscores(panel_rows(spreads, premiums), StudyConfig(zscore_window_days=150))
        summary = signal_summary(rows)
        assert [row["signal"] for row in summary] == [
            SHORT_CORRELATION,
            NO_TRADE,
            LONG_CORRELATION,
            "All days",
        ]

    def test_the_unconditional_row_counts_every_measurable_day(self) -> None:
        rng = np.random.default_rng(3)
        spreads = list(rng.normal(0.0, 0.02, 200))
        premiums = list(rng.normal(0.0, 0.05, 200))
        rows = add_zscores(panel_rows(spreads, premiums), StudyConfig(zscore_window_days=100))
        summary = signal_summary(rows)
        total = next(row for row in summary if row["signal"] == "All days")
        assert total["n"] == len(rows)

    def test_states_partition_the_measurable_days(self) -> None:
        rng = np.random.default_rng(4)
        spreads = list(rng.normal(0.0, 0.02, 250))
        premiums = list(rng.normal(0.0, 0.05, 250))
        rows = add_zscores(panel_rows(spreads, premiums), StudyConfig(zscore_window_days=120))
        summary = {row["signal"]: row["n"] for row in signal_summary(rows)}
        assert (
            summary[SHORT_CORRELATION] + summary[NO_TRADE] + summary[LONG_CORRELATION]
            == summary["All days"]
        )

    def test_an_empty_state_reports_zero_not_a_crash(self) -> None:
        rows = add_zscores(panel_rows([0.0] * 200, [0.01] * 200), StudyConfig())
        summary = {row["signal"]: row["n"] for row in signal_summary(rows)}
        assert summary[SHORT_CORRELATION] == 0


class TestRegimes:
    def test_boundaries_are_the_terciles(self) -> None:
        rows: list[dict[str, Any]] = [{"index_iv": 0.10 + 0.001 * index} for index in range(300)]
        lower, upper = regime_boundaries(rows)
        levels = np.array([row["index_iv"] for row in rows])
        assert lower == pytest.approx(float(np.quantile(levels, 1 / 3)))
        assert upper == pytest.approx(float(np.quantile(levels, 2 / 3)))

    def test_classification_matches_the_boundaries(self) -> None:
        bounds = (0.15, 0.25)
        assert volatility_regime(0.10, bounds) == "Low volatility"
        assert volatility_regime(0.15, bounds) == "Low volatility"
        assert volatility_regime(0.20, bounds) == "Middle volatility"
        assert volatility_regime(0.25, bounds) == "Middle volatility"
        assert volatility_regime(0.40, bounds) == "High volatility"

    def test_missing_levels_are_ignored_when_forming_boundaries(self) -> None:
        rows: list[dict[str, Any]] = [{"index_iv": None}, {"index_iv": 0.1}, {"index_iv": 0.3}]
        lower, upper = regime_boundaries(rows)
        assert lower < upper

    def test_no_levels_at_all_is_refused(self) -> None:
        with pytest.raises(ValueError, match="No index implied-volatility"):
            regime_boundaries([{"index_iv": None}])
