"""Input parsing, including the two shapes Cboe actually publishes."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pytest

from dispersion.loaders import (
    load_basket,
    load_cboe_history,
    load_price_history,
    load_price_series,
    parse_bool,
    resolve_value_column,
)

from .conftest import (
    synthetic_basket,
    synthetic_prices,
    trading_dates,
    write_basket_csv,
    write_cboe_csv,
    write_price_csv,
)


class TestParseBool:
    @pytest.mark.parametrize("value", ["TRUE", "true", "1", "yes", "Y", " true "])
    def test_truthy_spellings(self, value: str) -> None:
        assert parse_bool(value) is True

    @pytest.mark.parametrize("value", ["FALSE", "false", "0", "no", "", None])
    def test_everything_else_is_false(self, value: str | None) -> None:
        assert parse_bool(value) is False


class TestResolveValueColumn:
    def test_prefers_close_when_present(self) -> None:
        assert resolve_value_column(["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]) == "CLOSE"

    def test_falls_back_to_the_single_remaining_column(self) -> None:
        # Cboe's dispersion index names its value column after the index itself rather than CLOSE.
        assert resolve_value_column(["DATE", "DSPX"]) == "DSPX"

    def test_a_missing_date_column_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="No DATE column"):
            resolve_value_column(["WHEN", "CLOSE"])

    def test_an_ambiguous_header_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Cannot identify the value column"):
            resolve_value_column(["DATE", "ONE", "TWO"])


class TestLoadCboeHistory:
    def test_reads_the_five_column_shape(self, tmp_path: Path) -> None:
        calendar = trading_dates(date(2013, 1, 2), 3)
        write_cboe_csv(tmp_path / "COR1M_History.csv", calendar, [30.0, 35.0, 40.0])
        levels = load_cboe_history(tmp_path / "COR1M_History.csv")
        assert list(levels.values()) == pytest.approx([0.30, 0.35, 0.40])

    def test_reads_the_two_column_shape(self, tmp_path: Path) -> None:
        calendar = trading_dates(date(2014, 6, 19), 2)
        write_cboe_csv(tmp_path / "DSPX_History.csv", calendar, [17.81, 17.37], value_column="DSPX")
        levels = load_cboe_history(tmp_path / "DSPX_History.csv")
        assert list(levels.values()) == pytest.approx([0.1781, 0.1737])

    def test_converts_percentage_points_to_decimals(self, tmp_path: Path) -> None:
        write_cboe_csv(tmp_path / "VIX_History.csv", trading_dates(date(2013, 1, 2), 1), [20.0])
        assert next(
            iter(load_cboe_history(tmp_path / "VIX_History.csv").values())
        ) == pytest.approx(0.20)

    def test_a_nonpositive_level_is_treated_as_missing_and_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Cboe's dispersion index carries a literal 0.0 on 8 February 2018 between values of 14.05
        # and 20.92. That is a missing print, not an observation, and rejecting the whole file over
        # it would make the study un-runnable on real published data.
        calendar = trading_dates(date(2013, 1, 2), 3)
        write_cboe_csv(
            tmp_path / "DSPX_History.csv", calendar, [14.05, 0.0, 20.92], value_column="DSPX"
        )
        with caplog.at_level(logging.WARNING):
            levels = load_cboe_history(tmp_path / "DSPX_History.csv")
        assert len(levels) == 2
        assert calendar[1] not in levels
        assert "nonpositive" in caplog.text

    def test_a_file_of_only_bad_prints_is_rejected(self, tmp_path: Path) -> None:
        calendar = trading_dates(date(2013, 1, 2), 2)
        write_cboe_csv(tmp_path / "DSPX_History.csv", calendar, [0.0, 0.0], value_column="DSPX")
        with pytest.raises(ValueError, match="No data rows"):
            load_cboe_history(tmp_path / "DSPX_History.csv")

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Missing Cboe input"):
            load_cboe_history(tmp_path / "absent.csv")

    def test_unusable_header(self, tmp_path: Path) -> None:
        path = tmp_path / "x.csv"
        path.write_text("WHEN,CLOSE\n01/02/2013,20\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No DATE column"):
            load_cboe_history(path)

    def test_unparseable_date(self, tmp_path: Path) -> None:
        path = tmp_path / "x.csv"
        path.write_text("DATE,CLOSE\n2013-01-02,20\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid row"):
            load_cboe_history(path)

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "x.csv"
        path.write_text("DATE,CLOSE\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No data rows"):
            load_cboe_history(path)


class TestLoadPriceHistory:
    def test_reads_a_two_column_file(self, tmp_path: Path) -> None:
        calendar = trading_dates(date(2013, 1, 2), 3)
        write_price_csv(
            tmp_path / "AAPL_prices.csv", dict(zip(calendar, [10.0, 11.0, 12.0], strict=True))
        )
        assert load_price_history(tmp_path / "AAPL_prices.csv")[calendar[1]] == 11.0

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Missing price input"):
            load_price_history(tmp_path / "absent.csv")

    def test_nonpositive_close_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "p.csv"
        path.write_text("date,close\n2013-01-02,0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Nonpositive close"):
            load_price_history(path)

    def test_missing_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "p.csv"
        path.write_text("date,open\n2013-01-02,10\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected columns"):
            load_price_history(path)


class TestLoadBasket:
    def test_reads_constituents_and_their_tradeability(self, tmp_path: Path) -> None:
        basket = synthetic_basket(8)
        write_basket_csv(tmp_path / "basket.csv", basket)
        loaded = load_basket(tmp_path / "basket.csv")
        assert [item.ticker for item in loaded] == [item.ticker for item in basket]
        assert sum(item.in_iv_basket for item in loaded) == sum(
            item.in_iv_basket for item in basket
        )

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Missing basket reference"):
            load_basket(tmp_path / "absent.csv")

    def test_missing_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "basket.csv"
        path.write_text("ticker,name\nAAPL,Apple\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain"):
            load_basket(path)

    def test_empty_basket_is_rejected(self, tmp_path: Path) -> None:
        write_basket_csv(tmp_path / "basket.csv", [])
        with pytest.raises(ValueError, match="No constituents found"):
            load_basket(tmp_path / "basket.csv")


class TestLoadPriceSeries:
    def test_loads_the_index_and_every_constituent(self, tmp_path: Path) -> None:
        basket = synthetic_basket(6)
        calendar = trading_dates(date(2013, 1, 2), 40)
        series = synthetic_prices(basket, calendar)
        for ticker, history in series.items():
            write_price_csv(tmp_path / f"{ticker}_prices.csv", history)
        loaded = load_price_series(tmp_path, basket)
        assert set(loaded) == {"INDEX", *[item.ticker for item in basket]}

    def test_a_missing_constituent_file_is_reported(self, tmp_path: Path) -> None:
        basket = synthetic_basket(3)
        calendar = trading_dates(date(2013, 1, 2), 30)
        write_price_csv(tmp_path / "INDEX_prices.csv", synthetic_prices(basket, calendar)["INDEX"])
        with pytest.raises(FileNotFoundError):
            load_price_series(tmp_path, basket)
