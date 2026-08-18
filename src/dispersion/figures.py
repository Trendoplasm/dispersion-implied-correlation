"""The study's four figures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib

# A non-interactive backend keeps the study runnable headless, in CI and over SSH.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dispersion.config import REGIME_ORDER
from dispersion.models import Row
from dispersion.signals import LONG_CORRELATION, SHORT_CORRELATION

FIGURE_DPI = 180


def _finish(
    figure: Figure,
    axis: Axes,
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    xlabel: str | None = None,
    grid_axis: Literal["both", "x", "y"] = "both",
    legend: bool = True,
) -> None:
    """Apply the shared styling and write the file."""
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if legend:
        axis.legend()
    axis.grid(axis=grid_axis, alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def plot_correlation_history(panel: Sequence[Row], output_path: Path) -> None:
    """Plot implied correlation against what the basket subsequently delivered.

    The two lines track each other closely, which is the study's first finding: on average the
    option market prices constituent co-movement about right.
    """
    dates = [row["date"] for row in panel]
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        dates,
        [100 * row["implied_correlation"] for row in panel],
        linewidth=1.0,
        label="Implied correlation (Cboe COR1M)",
    )
    forward = [
        (row["date"], 100 * row["forward_realized_correlation"])
        for row in panel
        if row["forward_realized_correlation"] is not None
    ]
    axis.plot(
        [d for d, _ in forward],
        [v for _, v in forward],
        linewidth=1.0,
        alpha=0.8,
        label="Realised correlation over the following month",
    )
    _finish(
        figure,
        axis,
        output_path,
        title="Implied Versus Subsequently Realised Correlation",
        xlabel="Date",
        ylabel="Average constituent correlation (%)",
    )


def plot_premium_by_signal(signal_rows: Sequence[Row], output_path: Path) -> None:
    """Plot the realised premium conditional on the signal state.

    This is the study's central result: unconditionally the premium is nil, but it is clearly
    present after a short-correlation signal and clearly negative after a long-correlation one.
    """
    order = [SHORT_CORRELATION, "no_trade", LONG_CORRELATION, "All days"]
    labels = {
        SHORT_CORRELATION: "Short-correlation\nsignal",
        "no_trade": "No signal",
        LONG_CORRELATION: "Long-correlation\nsignal",
        "All days": "All days",
    }
    rows = [next((r for r in signal_rows if r["signal"] == state), None) for state in order]
    present = [
        (labels[state], row) for state, row in zip(order, rows, strict=True) if row and row["n"]
    ]

    positions = np.arange(len(present))
    figure, axis = plt.subplots(figsize=(9, 6))
    values = [100 * row["mean_premium"] for _, row in present]
    axis.bar(positions, values, 0.55, label="Mean premium")
    for position, (_, row) in enumerate(present):
        axis.annotate(
            f"n={row['n']}\n{row['pct_positive_premium']:.0%} positive",
            (position, values[position]),
            textcoords="offset points",
            xytext=(0, 6 if values[position] >= 0 else -26),
            ha="center",
            fontsize=8,
        )
    axis.axhline(0, linewidth=0.8, color="black")
    axis.set_xticks(positions, [label for label, _ in present])
    _finish(
        figure,
        axis,
        output_path,
        title="The Correlation Premium Is Conditional, Not Constant",
        ylabel="Implied minus realised correlation (percentage points)",
        grid_axis="y",
    )


def plot_attribution(attribution: Sequence[Row], output_path: Path) -> None:
    """Stack where each direction's profit came from."""
    groups = [row["group"] for row in attribution]
    positions = np.arange(len(groups))
    components = [
        ("mean_correlation_pnl", "Correlation (volatility dispersion)"),
        ("mean_volatility_pnl", "Volatility (average level)"),
        ("mean_residual_pnl", "Residual (gamma, decay, hedging)"),
    ]
    figure, axis = plt.subplots(figsize=(10, 6))
    positive_base = np.zeros(len(groups))
    negative_base = np.zeros(len(groups))
    for key, label in components:
        values = np.array([row[key] for row in attribution], dtype=float)
        base = np.where(values >= 0, positive_base, negative_base)
        axis.bar(positions, values, 0.5, bottom=base, label=label)
        positive_base = positive_base + np.where(values >= 0, values, 0.0)
        negative_base = negative_base + np.where(values < 0, values, 0.0)
    costs = -np.array([row["mean_cost"] for row in attribution], dtype=float)
    axis.bar(positions, costs, 0.5, bottom=negative_base, label="Transaction costs")
    axis.plot(
        positions,
        [row["mean_net_pnl"] for row in attribution],
        "kD",
        markersize=8,
        label="Net profit",
    )
    axis.axhline(0, linewidth=0.8, color="black")
    axis.set_xticks(positions, [g.replace("_", " ") for g in groups])
    _finish(
        figure,
        axis,
        output_path,
        title="Where Dispersion Profit Comes From",
        ylabel="Mean dollars per trade",
        grid_axis="y",
    )


def plot_regime_premium(regimes: Sequence[Row], output_path: Path) -> None:
    """Plot the premium and its hit rate by volatility regime."""
    present = [row for row in regimes if row["regime"] in REGIME_ORDER]
    labels = [row["regime"] for row in present]
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.bar(
        positions - width / 2,
        [100 * row["mean_implied_correlation"] for row in present],
        width,
        label="Mean implied correlation",
    )
    axis.bar(
        positions + width / 2,
        [100 * row["mean_forward_realized_correlation"] for row in present],
        width,
        label="Mean realised correlation",
    )
    for position, row in enumerate(present):
        axis.annotate(
            f"premium {100 * row['mean_premium']:+.1f}",
            (
                position,
                100
                * max(row["mean_implied_correlation"], row["mean_forward_realized_correlation"]),
            ),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )
    axis.set_xticks(positions, labels)
    _finish(
        figure,
        axis,
        output_path,
        title="Correlation Pricing by Volatility Regime",
        ylabel="Average constituent correlation (%)",
        grid_axis="y",
    )
