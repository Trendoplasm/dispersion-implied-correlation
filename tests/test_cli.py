"""Command-line behaviour, including how failures are reported."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from dispersion import __version__
from dispersion.cli import main

from .conftest import (
    TRADEABLE,
    cboe_series,
    synthetic_basket,
    synthetic_prices,
    trading_dates,
    write_basket_csv,
    write_cboe_csv,
    write_price_csv,
)

EXPECTED_TABLES = 13


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Build a complete synthetic input tree."""
    calendar = trading_dates(date(2013, 1, 2), 250)
    basket = synthetic_basket()
    prices = synthetic_prices(basket, calendar)
    raw = tmp_path / "raw"

    # A drifting implied correlation, so the z-score signal has something to fire on.
    series = cboe_series(calendar)
    drifting = {day: 0.45 + 0.10 * ((index // 25) % 2) for index, day in enumerate(calendar)}
    series["COR1M"] = drifting
    for name, history in series.items():
        column = "DSPX" if name == "DSPX" else "CLOSE"
        write_cboe_csv(
            raw / f"{name}_History.csv",
            calendar,
            [100 * history[day] for day in calendar],
            value_column=column,
        )
    for ticker, history in prices.items():
        write_price_csv(raw / f"{ticker}_prices.csv", history)
    write_basket_csv(tmp_path / "basket.csv", basket)
    return tmp_path


def run(workspace: Path, *extra: str) -> int:
    return main(
        [
            "--data-dir",
            str(workspace / "raw"),
            "--basket",
            str(workspace / "basket.csv"),
            "--output-dir",
            str(workspace / "out"),
            "--start-date",
            "2013-01-02",
            "--bootstrap-iterations",
            "50",
            "--warmup-days",
            "70",
            *extra,
        ]
    )


def test_writes_every_table_and_figure(workspace: Path) -> None:
    assert run(workspace) == 0
    out = workspace / "out"
    assert len(list((out / "tables").glob("*.csv"))) == EXPECTED_TABLES
    assert {p.name for p in (out / "plots").glob("*.png")} == {
        "correlation_history.png",
        "premium_by_signal.png",
        "pnl_attribution.png",
        "regime_premium.png",
    }
    assert (out / "summary.json").exists()


def test_reports_the_headline_result(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(workspace, "--no-plots")
    output = capsys.readouterr().out
    assert "dispersion trades" in output
    assert "correlation premium" in output


def test_summary_states_the_scope(workspace: Path) -> None:
    run(workspace, "--no-plots")
    payload = json.loads((workspace / "out" / "summary.json").read_text())
    # Every export restates the two limits on what the backtest can claim.
    assert "five-name proxy" in payload["scope_note"]
    assert "not broker margin" in payload["scope_note"]
    assert payload["tradeable_constituents"] == list(TRADEABLE)


def test_cost_stress_includes_a_baseline_and_harsher_cases(workspace: Path) -> None:
    run(workspace, "--no-plots")
    payload = json.loads((workspace / "out" / "summary.json").read_text())
    stress = payload["cost_stress"]
    assert any(row["is_baseline"] for row in stress)
    multiples = [row["cost_multiple"] for row in stress]
    assert multiples == sorted(multiples)
    # Harsher costs must never look cheaper.
    costs = [row["mean_cost"] for row in stress]
    assert costs == sorted(costs)


def test_no_plots_skips_figures(workspace: Path) -> None:
    assert run(workspace, "--no-plots") == 0
    assert not (workspace / "out" / "plots").exists()


def test_quiet_suppresses_progress_output(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run(workspace, "--quiet", "--no-plots")
    assert capsys.readouterr().out == ""


def test_holding_period_is_configurable(workspace: Path) -> None:
    import csv

    assert run(workspace, "--no-plots", "--holding-days", "15") == 0
    rows = list(csv.DictReader((workspace / "out" / "tables" / "dispersion_trades.csv").open()))
    assert rows
    for row in rows:
        entry = date.fromisoformat(row["entry_date"])
        exit_ = date.fromisoformat(row["exit_date"])
        # 15 trading days spans at most three calendar weeks.
        assert (exit_ - entry).days <= 25


def test_too_short_a_holding_period_is_refused_with_an_explanation(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The realised leg is measured over the holding period, so a period shorter than the minimum
    # estimation window would leave the premium unmeasurable. Better to say that than to fail deep
    # inside the summary code.
    status = run(workspace, "--no-plots", "--holding-days", "5")
    assert status == 2
    assert "must be at least" in capsys.readouterr().err


def test_missing_input_reports_a_message_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        ["--data-dir", str(tmp_path / "absent"), "--basket", str(tmp_path / "absent.csv")]
    )
    captured = capsys.readouterr()
    assert status == 1
    assert captured.err.lower().startswith("error:")
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("flag", ["--bootstrap-iterations", "--holding-days"])
def test_nonpositive_settings_are_usage_errors(
    workspace: Path, capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    status = main(
        [
            "--data-dir",
            str(workspace / "raw"),
            "--basket",
            str(workspace / "basket.csv"),
            "--output-dir",
            str(workspace / "out"),
            flag,
            "0",
        ]
    )
    assert status == 2
    assert "must be positive" in capsys.readouterr().err


def test_a_signal_that_never_fires_is_reported_not_crashed(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A warm-up longer than the sample legitimately yields no trades. The run should still complete
    # and say so, because "no trades" is a result.
    status = main(
        [
            "--data-dir",
            str(workspace / "raw"),
            "--basket",
            str(workspace / "basket.csv"),
            "--output-dir",
            str(workspace / "empty"),
            "--start-date",
            "2013-01-02",
            "--bootstrap-iterations",
            "50",
            "--warmup-days",
            "100000",
            "--no-plots",
        ]
    )
    assert status == 0
    assert "no trades taken" in capsys.readouterr().out
    payload = json.loads((workspace / "empty" / "summary.json").read_text())
    assert payload["trades"] == 0
    # The panel still exists, because measuring the premium never depended on trading it.
    assert (workspace / "empty" / "tables" / "correlation_panel.csv").exists()
    assert not (workspace / "empty" / "tables" / "dispersion_trades.csv").exists()


def test_negative_warmup_is_a_usage_error(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        [
            "--data-dir",
            str(workspace / "raw"),
            "--basket",
            str(workspace / "basket.csv"),
            "--output-dir",
            str(workspace / "out"),
            "--warmup-days",
            "-1",
        ]
    )
    assert status == 2
    assert "cannot be negative" in capsys.readouterr().err


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out
