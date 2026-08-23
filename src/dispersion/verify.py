"""Numeric comparison of two output directories.

Reproducing this study bit-for-bit is not achievable, and not a meaningful goal: the ordinary
least-squares routines underneath depend on the host's linear-algebra library, so identical code
on two machines can disagree in the last floating-point digit. Observed drift between macOS and
Linux is on the order of 1e-13 relative.

Two values are therefore treated as agreeing when

.. code-block:: text

    |a - b| <= atol + rtol * max(|a|, |b|)

which is the combined absolute-and-relative test :func:`numpy.isclose` uses. Both halves are
needed. A purely relative test cannot work here, because several exported columns are the
*residual of an identity* whose correct value is zero: ``attribution_error`` and ``check`` report
how far the leg-by-leg decomposition missed the realised profit, and a correct run puts them at
1e-12. Comparing 1e-12 against 0.0 relatively gives a difference of 100%, so a relative-only
check calls a perfectly reproduced study a failure. The absolute floor is what makes those
columns comparable, and it sits many orders of magnitude below the smallest quantity the study
reports.
"""

from __future__ import annotations

import csv
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

#: Relative tolerance treated as agreement. Several orders of magnitude above the observed
#: cross-platform noise floor, and many orders below anything that could change a reported
#: statistic.
DEFAULT_RTOL = 1e-9

#: Absolute tolerance treated as agreement. Set by the identity-residual columns described in the
#: module docstring, whose largest observed cross-platform drift is 1.8e-12. The smallest quantity
#: the study actually reports is around 1e-3, so this floor cannot mask a real difference.
DEFAULT_ATOL = 1e-10

#: Per-column tolerance overrides, as ``column`` or ``file.csv:column`` -> ``(rtol, atol)``.
#: Empty here: every column in this study is well determined at the defaults.
COLUMN_TOLERANCES: Mapping[str, tuple[float, float]] = {}


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing two output directories.

    Attributes:
        max_relative_difference: Largest relative difference found in any numeric field.
        max_difference_field: ``file:column`` where that difference occurred.
        discrepancies: Human-readable descriptions of every field outside tolerance, and of any
            structural mismatch such as a missing file or differing columns.
        compared_files: Number of table files compared.
    """

    max_relative_difference: float
    max_difference_field: str
    discrepancies: list[str]
    compared_files: int

    @property
    def matches(self) -> bool:
        """Whether the two directories agree within tolerance."""
        return not self.discrepancies


def _read_table(path: Path) -> list[dict[str, str]]:
    """Read one CSV table as a list of string-valued rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tolerances(
    file_name: str,
    column: str,
    rtol: float,
    atol: float,
    overrides: Mapping[str, tuple[float, float]],
) -> tuple[float, float]:
    """Return the tolerance pair for one column, the most specific override winning."""
    qualified = f"{file_name}:{column}"
    if qualified in overrides:
        return overrides[qualified]
    if column in overrides:
        return overrides[column]
    return rtol, atol


def _compare_cell(left: str, right: str, rtol: float, atol: float) -> tuple[bool, float] | None:
    """Compare two cells numerically.

    Args:
        left: Reference cell.
        right: Candidate cell.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        ``(agrees, relative_difference)``, or None if the cells are not both numeric.
    """
    try:
        a, b = float(left), float(right)
    except ValueError:
        return None
    scale = max(abs(a), abs(b))
    difference = abs(a - b)
    agrees = difference <= atol + rtol * scale
    return agrees, (difference / scale if scale > 0 else 0.0)


def compare_output_dirs(
    expected: Path,
    actual: Path,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    ignore: Collection[str] = (),
    column_tolerances: Mapping[str, tuple[float, float]] | None = None,
) -> ComparisonResult:
    """Compare the ``tables/`` CSVs of two output directories.

    Args:
        expected: Reference output directory.
        actual: Directory to check.
        rtol: Relative tolerance treated as agreement.
        atol: Absolute tolerance treated as agreement.
        ignore: File names to leave out of the comparison. Use this only for tables that are
            passthroughs of the raw inputs, whose length depends on the vintage of the
            downloaded data rather than on anything the study computes.
        column_tolerances: Per-column tolerance overrides. Defaults to :data:`COLUMN_TOLERANCES`.

    Returns:
        The comparison outcome.

    Raises:
        FileNotFoundError: If the reference directory has no ``tables/`` subdirectory.
    """
    overrides = COLUMN_TOLERANCES if column_tolerances is None else column_tolerances
    expected_tables = expected / "tables"
    if not expected_tables.is_dir():
        raise FileNotFoundError(f"No tables directory to compare against: {expected_tables}")

    discrepancies: list[str] = []
    worst = 0.0
    worst_field = "none"
    files = [p for p in sorted(expected_tables.glob("*.csv")) if p.name not in ignore]

    for reference_file in files:
        candidate_file = actual / "tables" / reference_file.name
        if not candidate_file.exists():
            discrepancies.append(f"{reference_file.name}: missing from {actual}")
            continue

        reference_rows = _read_table(reference_file)
        candidate_rows = _read_table(candidate_file)
        if len(reference_rows) != len(candidate_rows):
            discrepancies.append(
                f"{reference_file.name}: {len(reference_rows)} rows expected, "
                f"{len(candidate_rows)} found"
            )
            continue
        if reference_rows and list(reference_rows[0]) != list(candidate_rows[0]):
            discrepancies.append(f"{reference_file.name}: column names differ")
            continue

        for number, (reference, candidate) in enumerate(
            zip(reference_rows, candidate_rows, strict=True), start=2
        ):
            for column, reference_value in reference.items():
                candidate_value = candidate[column]
                if reference_value == candidate_value:
                    continue
                cell_rtol, cell_atol = _tolerances(
                    reference_file.name, column, rtol, atol, overrides
                )
                outcome = _compare_cell(reference_value, candidate_value, cell_rtol, cell_atol)
                if outcome is None:
                    discrepancies.append(
                        f"{reference_file.name} line {number} [{column}]: "
                        f"{reference_value!r} expected, {candidate_value!r} found"
                    )
                    continue
                agrees, relative = outcome
                if relative > worst:
                    worst, worst_field = relative, f"{reference_file.name}:{column}"
                if not agrees:
                    discrepancies.append(
                        f"{reference_file.name} line {number} [{column}]: "
                        f"{reference_value} expected, {candidate_value} found "
                        f"(relative difference {relative:.2e})"
                    )

    return ComparisonResult(
        max_relative_difference=worst,
        max_difference_field=worst_field,
        discrepancies=discrepancies,
        compared_files=len(files),
    )
