"""Small numeric helpers shared by the aggregation, panel, and signal stages.

These exist so an empty sample yields a visibly missing value rather than a NaN. A NaN
propagates silently through a CSV into a spreadsheet cell that looks like a number; None
appears as an empty cell that a reader has to account for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np

#: A sample of measurements, held either as a plain sequence or as an array.
Samples: TypeAlias = Sequence[float] | np.ndarray


def mean_or_none(values: Samples) -> float | None:
    """Return the sample mean, or None if there is nothing to average."""
    return float(np.mean(values)) if len(values) else None


def median_or_none(values: Samples) -> float | None:
    """Return the sample median, or None if there is nothing to rank."""
    return float(np.median(values)) if len(values) else None


def std_dev_or_none(values: Samples) -> float | None:
    """Return the sample standard deviation, or None if it is undefined.

    A single observation has no sample dispersion; ``ddof=1`` would divide by zero.
    """
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def present(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    """Collect the non-missing values of one field across rows."""
    return [row[key] for row in rows if row.get(key) is not None]
