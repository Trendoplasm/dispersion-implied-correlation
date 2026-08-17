"""Serialisers for the study's tables and JSON summary."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np


def json_default(value: Any) -> Any:
    """Serialise the non-JSON-native types the study produces.

    Args:
        value: Object json cannot encode.

    Returns:
        A JSON-encodable equivalent.

    Raises:
        TypeError: If the type is not one the study is expected to emit.
    """
    if isinstance(value, date | datetime):
        return value.isoformat()
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a table to CSV, taking column order from the first row.

    Args:
        path: Destination file; parent directories are created.
        rows: Records to write. Every row is expected to share the first row's keys; extra
            keys are ignored rather than shifting later columns.

    Raises:
        ValueError: If there are no rows. An empty table means an upstream failure, and a
            header-only file would hide it.
    """
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (value.isoformat() if isinstance(value, date | datetime) else value)
                    for key, value in row.items()
                }
            )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write the run summary as indented JSON.

    Args:
        path: Destination file; parent directories are created.
        payload: Summary object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)
