"""End-to-end reproduction of the published results.

This is the test that guards the numbers. It runs the complete study against the downloaded
histories and compares every exported table with the committed results.

It skips itself when the inputs are absent, which is the case on a fresh clone and in continuous
integration, because neither Cboe's index history nor the price history is redistributed here.
Populate them with ``python scripts/fetch_cboe_data.py`` and ``python scripts/fetch_price_data.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dispersion.config import CostModel, StudyConfig
from dispersion.pipeline import StudyResults, headline, run_study, write_outputs
from dispersion.signals import LONG_CORRELATION, SHORT_CORRELATION
from dispersion.trade import attribution_error
from dispersion.verify import compare_output_dirs

EXPECTED_HEADLINE = (
    "Completed 124 dispersion trades over 3372 panel days: unconditional correlation premium "
    "-0.17 points, +3.98 points after a short-correlation signal, mean trade return -0.07% on "
    "research capital."
)

EXPECTED_PANEL_OBSERVATIONS = 3393
EXPECTED_TRADES = 124
EXPECTED_TABLE_COUNT = 13
EXPECTED_TRADEABLE = ["AAPL", "AMZN", "GOOGL", "GS", "IBM"]

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "raw"
    basket = root / "data" / "reference" / "basket.csv"
    if not data_dir.is_dir() or not basket.exists():
        pytest.skip("downloaded inputs absent; run the fetch scripts to enable")
    return data_dir, basket


@pytest.fixture(scope="module")
def results(paths: tuple[Path, Path]) -> StudyResults:
    data_dir, basket = paths
    return run_study(data_dir, basket, StudyConfig(), CostModel())


def test_headline_result_is_unchanged(results: StudyResults) -> None:
    assert headline(results) == EXPECTED_HEADLINE


def test_sample_sizes(results: StudyResults) -> None:
    assert len(results.panel) == EXPECTED_PANEL_OBSERVATIONS
    assert len(results.trades) == EXPECTED_TRADES


def test_only_names_with_a_volatility_index_are_traded(results: StudyResults) -> None:
    tradeable = [item.ticker for item in results.basket if item.in_iv_basket]
    assert tradeable == EXPECTED_TRADEABLE
    # The realised-correlation basket is much wider than the traded one, and that gap is the
    # study's central limitation rather than an oversight.
    assert len(results.basket) > 40


def test_there_is_no_unconditional_correlation_premium(results: StudyResults) -> None:
    # The study's first finding, and a negative one: on average the option market prices
    # constituent co-movement about right.
    pooled = results.pooled_panel
    assert abs(pooled["mean_correlation_premium"]) < 0.01
    interval = results.bootstrap["correlation_premium"]
    assert interval is not None
    low, high, _ = interval
    assert low < 0 < high


def test_the_signal_identifies_a_premium_that_is_there(results: StudyResults) -> None:
    # The study's central finding: conditional on a high z-score the premium is clearly positive,
    # and the bootstrap interval excludes zero.
    short_signal = next(row for row in results.signal_summary if row["signal"] == SHORT_CORRELATION)
    long_signal = next(row for row in results.signal_summary if row["signal"] == LONG_CORRELATION)
    assert short_signal["mean_premium"] > 0.02
    assert long_signal["mean_premium"] < 0
    interval = results.bootstrap["short_signal_premium"]
    assert interval is not None
    low, high, _ = interval
    assert 0 < low < high


def test_the_premium_rises_with_volatility(results: StudyResults) -> None:
    regimes = {row["regime"]: row for row in results.premium_regimes}
    assert regimes["Low volatility"]["mean_premium"] < regimes["High volatility"]["mean_premium"]


def test_the_structure_is_vega_neutral_in_practice(results: StudyResults) -> None:
    # Volatility P&L is the part driven by the average level of implied volatility. On a
    # vega-neutral structure it should be negligible next to the correlation term.
    pooled = results.attribution[0]
    assert pooled["volatility_share_of_gross"] < 0.10


def test_every_trade_satisfies_the_attribution_identity(results: StudyResults) -> None:
    assert max(attribution_error(trade) for trade in results.trades) < 1e-6


def test_costs_are_material_relative_to_the_edge(results: StudyResults) -> None:
    # The honest bottom line: raising costs turns the strategy negative, so the edge and the
    # transaction costs are of comparable size.
    stress = {row["cost_multiple"]: row for row in results.cost_stress}
    assert stress[0.5]["mean_return_on_capital"] > stress[2.0]["mean_return_on_capital"]
    assert stress[3.0]["mean_return_on_capital"] < 0


def test_the_frozen_end_date_bounds_the_sample(results: StudyResults) -> None:
    cutoff = results.config.end()
    assert max(row["date"] for row in results.panel) <= cutoff
    assert max(trade.exit_date for trade in results.trades) <= cutoff


def test_all_tables_match_the_committed_results(results: StudyResults, tmp_path: Path) -> None:
    committed = Path(__file__).resolve().parent.parent / "outputs"
    if not (committed / "tables").is_dir():
        pytest.skip("no committed outputs to compare against")

    write_outputs(results, tmp_path, with_plots=False)
    comparison = compare_output_dirs(committed, tmp_path)
    assert comparison.matches, "\n".join(comparison.discrepancies[:20])
    assert comparison.compared_files == EXPECTED_TABLE_COUNT
