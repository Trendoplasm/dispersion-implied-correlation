"""End-to-end orchestration: load, measure, signal, backtest, summarise, export."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dispersion.aggregate import (
    attribution_table,
    bootstrap_interval,
    period_table,
    premium_by_regime,
    regime_table,
    strategy_summaries,
    summarize_trades,
    tail_tables,
)
from dispersion.config import (
    CONSTITUENT_VOLATILITY_INDEXES,
    INDEX_TICKER,
    INDEX_VOLATILITY_INDEX,
    CostModel,
    StudyConfig,
)
from dispersion.figures import (
    plot_attribution,
    plot_correlation_history,
    plot_premium_by_signal,
    plot_regime_premium,
)
from dispersion.loaders import load_basket, load_cboe_series, load_price_series
from dispersion.models import Constituent, LevelByDate, Row, Table
from dispersion.panel import BasketReturns, build_panel, panel_summary
from dispersion.signals import (
    LONG_CORRELATION,
    SHORT_CORRELATION,
    add_zscores,
    regime_boundaries,
    signal_summary,
)
from dispersion.trade import DispersionTrade, run_backtest, trade_row
from dispersion.writers import write_csv, write_json

logger = logging.getLogger(__name__)

#: Restated in every export so no downstream reader can mistake the study's scope.
SCOPE_NOTE = (
    "Implied correlation is Cboe's observed COR1M index and realised correlation is computed from "
    "observed closing prices, so the premium is measured entirely from market data. Two limits "
    "bound what the backtest can claim. First, only five S&P 500 constituents have a published "
    "volatility index, so the constituent leg of the traded structure is a five-name proxy for a "
    "500-name index rather than a replica of it. Second, option prices are Black-Scholes marks "
    "from 30-day at-the-money volatility indexes rather than historical quotes, so reported "
    "profits exclude the skew and the bid-ask depth of a real chain beyond a modelled entry "
    "half-spread. The capital denominator is a research normalisation, not broker margin."
)

#: Cost multiples used for the stress table.
COST_MULTIPLES: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)


@dataclass(frozen=True)
class StudyResults:
    """Everything one run produces."""

    config: StudyConfig
    costs: CostModel
    basket: list[Constituent] = field(repr=False)
    panel: Table = field(repr=False)
    panel_summaries: Table = field(repr=False)
    signal_summary: Table = field(repr=False)
    premium_regimes: Table = field(repr=False)
    trades: list[DispersionTrade] = field(repr=False)
    strategy_summaries: Table = field(repr=False)
    attribution: Table = field(repr=False)
    regimes: Table = field(repr=False)
    periods: Table = field(repr=False)
    cost_stress: Table = field(repr=False)
    worst_trades: Table = field(repr=False)
    best_trades: Table = field(repr=False)
    bootstrap: dict[str, tuple[float, float, float] | None] = field(repr=False)
    regime_bounds: tuple[float, float] = (0.0, 0.0)

    @property
    def pooled_panel(self) -> Row:
        """Return the all-days panel summary."""
        return self.panel_summaries[0]

    @property
    def pooled_strategy(self) -> Row | None:
        """Return the all-trades strategy summary, or None if no trade was taken."""
        return self.strategy_summaries[0] if self.strategy_summaries else None


def _volatility_series(cboe_series: dict[str, LevelByDate]) -> dict[str, LevelByDate]:
    """Map each tradeable ticker to its observed implied-volatility series."""
    return {INDEX_TICKER: cboe_series[INDEX_VOLATILITY_INDEX]} | {
        ticker: cboe_series[index] for ticker, index in CONSTITUENT_VOLATILITY_INDEXES.items()
    }


def run_study(
    data_dir: Path, basket_path: Path, config: StudyConfig, costs: CostModel
) -> StudyResults:
    """Load the inputs and run the complete study.

    Args:
        data_dir: Directory holding the downloaded histories.
        basket_path: Path to the reference basket CSV.
        config: Study windows, thresholds, and sizing.
        costs: Transaction-cost model.

    Returns:
        Every measurement, simulation, and summary the study produces.
    """
    cboe_series = load_cboe_series(data_dir)
    basket_reference = load_basket(basket_path)
    price_series = load_price_series(data_dir, basket_reference)
    basket = BasketReturns(basket_reference, price_series, config.start(), config.end())

    panel = add_zscores(build_panel(basket, cboe_series, config), config)
    bounds = regime_boundaries(panel)

    volatility_series = _volatility_series(cboe_series)
    tradeable = [item.ticker for item in basket_reference if item.in_iv_basket]
    trades = run_backtest(
        panel, basket.dates, price_series, volatility_series, tradeable, config, costs
    )

    panel_summaries: Table = [panel_summary(panel, "All days")]
    for label, predicate in (
        ("In sample", lambda row: row["in_sample"]),
        ("Out of sample", lambda row: not row["in_sample"]),
    ):
        subset = [row for row in panel if predicate(row)]
        if any(row["correlation_premium"] is not None for row in subset):
            panel_summaries.append(panel_summary(subset, label))

    # Cost stress: the same trades re-run under harsher and gentler assumptions. Costs are the
    # part of a dispersion trade most likely to be understated, because the structure holds many
    # legs and rebalances all of them daily.
    cost_stress: Table = []
    for multiple in COST_MULTIPLES:
        stressed = CostModel(
            option_commission=costs.option_commission * multiple,
            option_half_spread=costs.option_half_spread * multiple,
            index_hedge_cost_bps=costs.index_hedge_cost_bps * multiple,
            stock_hedge_cost_bps=costs.stock_hedge_cost_bps * multiple,
            contract_multiplier=costs.contract_multiplier,
        )
        stressed_trades = run_backtest(
            panel, basket.dates, price_series, volatility_series, tradeable, config, stressed
        )
        if not stressed_trades:
            continue
        summary = summarize_trades(stressed_trades, f"{multiple:g}x costs", config)
        summary["cost_multiple"] = multiple
        summary["is_baseline"] = multiple == 1.0
        cost_stress.append(summary)

    rng = np.random.default_rng(config.random_seed)
    premium_values = [
        row["correlation_premium"] for row in panel if row["correlation_premium"] is not None
    ]
    bootstrap = {
        "correlation_premium": bootstrap_interval(premium_values, rng, config.bootstrap_iterations),
        "short_signal_premium": bootstrap_interval(
            [
                row["correlation_premium"]
                for row in panel
                if row["signal"] == SHORT_CORRELATION and row["correlation_premium"] is not None
            ],
            rng,
            config.bootstrap_iterations,
        ),
        "trade_return_on_capital": bootstrap_interval(
            [trade.return_on_capital for trade in trades], rng, config.bootstrap_iterations
        ),
    }

    worst, best = tail_tables(trades)
    return StudyResults(
        config=config,
        costs=costs,
        basket=basket_reference,
        panel=panel,
        panel_summaries=panel_summaries,
        signal_summary=signal_summary(panel),
        premium_regimes=premium_by_regime(panel, bounds),
        trades=trades,
        strategy_summaries=strategy_summaries(trades, config),
        attribution=attribution_table(trades),
        regimes=regime_table(trades, bounds, config),
        periods=period_table(trades, config),
        cost_stress=cost_stress,
        worst_trades=worst,
        best_trades=best,
        bootstrap=bootstrap,
        regime_bounds=bounds,
    )


def summary_payload(results: StudyResults) -> dict[str, Any]:
    """Assemble the machine-readable run summary."""
    return {
        "configuration": asdict(results.config),
        "cost_model": asdict(results.costs),
        "basket_size": len(results.basket),
        "tradeable_constituents": [c.ticker for c in results.basket if c.in_iv_basket],
        "panel_observations": len(results.panel),
        "trades": len(results.trades),
        "regime_boundaries": list(results.regime_bounds),
        "panel_summaries": results.panel_summaries,
        "signal_summary": results.signal_summary,
        "premium_regimes": results.premium_regimes,
        "strategy_summaries": results.strategy_summaries,
        "attribution": results.attribution,
        "regimes": results.regimes,
        "periods": results.periods,
        "cost_stress": results.cost_stress,
        "bootstrap": results.bootstrap,
        "scope_note": SCOPE_NOTE,
    }


def write_outputs(results: StudyResults, output_dir: Path, *, with_plots: bool = True) -> None:
    """Write every table, figure, and summary of a completed run."""
    tables_dir = output_dir / "tables"
    tables: dict[str, Table] = {
        "correlation_panel": results.panel,
        "correlation_panel_summary": results.panel_summaries,
        "signal_summary": results.signal_summary,
        "premium_by_regime": results.premium_regimes,
        "dispersion_trades": [trade_row(trade) for trade in results.trades],
        "strategy_summary": results.strategy_summaries,
        "pnl_attribution": results.attribution,
        "trade_regimes": results.regimes,
        "in_out_of_sample": results.periods,
        "cost_stress": results.cost_stress,
        "worst_trades": results.worst_trades,
        "best_trades": results.best_trades,
        "reference_basket": [
            {
                "ticker": item.ticker,
                "name": item.name,
                "sector": item.sector,
                "iv_index": item.iv_index,
                "in_iv_basket": item.in_iv_basket,
            }
            for item in results.basket
        ],
    }
    empty = [name for name, rows in tables.items() if not rows]
    for name, rows in tables.items():
        if not rows:
            continue
        write_csv(tables_dir / f"{name}.csv", rows)
    if empty:
        # A signal that never fires, or a warm-up longer than the sample, legitimately yields no
        # trades. That is a result to report, not a failure to crash on -- and summary.json still
        # records the zero count, so the run is self-describing.
        logger.warning("No rows to write for: %s", ", ".join(sorted(empty)))

    if with_plots:
        plots_dir = output_dir / "plots"
        plot_correlation_history(results.panel, plots_dir / "correlation_history.png")
        plot_premium_by_signal(results.signal_summary, plots_dir / "premium_by_signal.png")
        if results.attribution:
            plot_attribution(results.attribution, plots_dir / "pnl_attribution.png")
        if results.premium_regimes:
            plot_regime_premium(results.premium_regimes, plots_dir / "regime_premium.png")

    write_json(output_dir / "summary.json", summary_payload(results))
    logger.info("Wrote %d tables to %s", len(tables) - len(empty), output_dir)


def headline(results: StudyResults) -> str:
    """Render the one-line result used to confirm a successful reproduction."""
    panel = results.pooled_panel
    short_signal = next(
        (row for row in results.signal_summary if row["signal"] == SHORT_CORRELATION), None
    )
    conditional = (
        f"{short_signal['mean_premium'] * 100:+.2f}"
        if short_signal and short_signal["n"]
        else "n/a"
    )
    strategy = results.pooled_strategy
    trade_return = (
        f"{strategy['mean_return_on_capital']:+.2%} on research capital"
        if strategy is not None
        else "no trades taken"
    )
    return (
        f"Completed {len(results.trades)} dispersion trades over {panel['n']} panel days: "
        f"unconditional correlation premium {panel['mean_correlation_premium'] * 100:+.2f} points, "
        f"{conditional} points after a short-correlation signal, mean trade return {trade_return}."
    )


#: Directions the study trades, re-exported for convenience.
DIRECTIONS = (SHORT_CORRELATION, LONG_CORRELATION)
