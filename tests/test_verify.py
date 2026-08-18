"""The tolerance-based output comparator."""

from __future__ import annotations

from pathlib import Path

import pytest

from dispersion.verify import DEFAULT_RTOL, compare_output_dirs


def write_table(directory: Path, name: str, text: str) -> None:
    tables = directory / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / name).write_text(text, encoding="utf-8")


BASE = "event_id,crush_pct\nAAPL_1,-0.1779266161910309\n"


def test_identical_directories_match(tmp_path: Path) -> None:
    for name in ("a", "b"):
        write_table(tmp_path / name, "t.csv", BASE)
    result = compare_output_dirs(tmp_path / "a", tmp_path / "b")
    assert result.matches
    assert result.compared_files == 1
    assert result.max_relative_difference == 0.0


def test_last_digit_drift_is_treated_as_agreement(tmp_path: Path) -> None:
    # This is the observed run-to-run noise floor of the underlying linear algebra.
    write_table(tmp_path / "a", "t.csv", BASE)
    write_table(tmp_path / "b", "t.csv", "event_id,crush_pct\nAAPL_1,-0.17792661619103095\n")
    result = compare_output_dirs(tmp_path / "a", tmp_path / "b")
    assert result.matches
    assert 0 < result.max_relative_difference < DEFAULT_RTOL


def test_a_real_difference_is_reported(tmp_path: Path) -> None:
    write_table(tmp_path / "a", "t.csv", BASE)
    write_table(tmp_path / "b", "t.csv", "event_id,crush_pct\nAAPL_1,-0.18\n")
    result = compare_output_dirs(tmp_path / "a", tmp_path / "b")
    assert not result.matches
    assert "crush_pct" in result.discrepancies[0]


def test_text_mismatch_is_reported(tmp_path: Path) -> None:
    write_table(tmp_path / "a", "t.csv", BASE)
    write_table(tmp_path / "b", "t.csv", "event_id,crush_pct\nAAPL_2,-0.1779266161910309\n")
    result = compare_output_dirs(tmp_path / "a", tmp_path / "b")
    assert not result.matches
    assert "event_id" in result.discrepancies[0]


def test_missing_file_is_reported(tmp_path: Path) -> None:
    write_table(tmp_path / "a", "t.csv", BASE)
    (tmp_path / "b" / "tables").mkdir(parents=True)
    result = compare_output_dirs(tmp_path / "a", tmp_path / "b")
    assert not result.matches
    assert "missing" in result.discrepancies[0]


def test_row_count_mismatch_is_reported(tmp_path: Path) -> None:
    write_table(tmp_path / "a", "t.csv", BASE + "AAPL_2,-0.2\n")
    write_table(tmp_path / "b", "t.csv", BASE)
    result = compare_output_dirs(tmp_path / "a", tmp_path / "b")
    assert not result.matches
    assert "rows" in result.discrepancies[0]


def test_column_change_is_reported(tmp_path: Path) -> None:
    write_table(tmp_path / "a", "t.csv", BASE)
    write_table(tmp_path / "b", "t.csv", "event_id,crush_percent\nAAPL_1,-0.1779266161910309\n")
    result = compare_output_dirs(tmp_path / "a", tmp_path / "b")
    assert not result.matches
    assert "column names differ" in result.discrepancies[0]


def test_missing_reference_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No tables directory"):
        compare_output_dirs(tmp_path / "absent", tmp_path)
