"""Numeric comparison of two output directories.

Reproducing this study bit-for-bit is not achievable, and not a meaningful goal: the ordinary
least-squares routines underneath depend on the host's linear-algebra library, so even two runs
of identical code on one machine can disagree in the last floating-point digit. Observed
run-to-run drift is on the order of 1e-14 relative.

Reproduction is therefore checked against a tolerance. :data:`DEFAULT_RTOL` sits several orders
of magnitude above that noise floor and many orders below anything that could change a reported
statistic, so it distinguishes "the same result" from "a real difference".
"""

from __future__ import annotations

import csv
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

#: Relative tolerance treated as agreement.
DEFAULT_RTOL = 1e-9

#: Floor on the comparison denominator, so values straddling zero compare sensibly.
_SCALE_FLOOR = 1e-12


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing two output directories.

    Attributes:
        max_relative_difference: Largest relative difference found in any numeric field.
        max_difference_field: ``file:column`` where that difference occurred.
        discrepancies: Human-readable descriptions of every field exceeding the tolerance, and
            of any structural mismatch such as a missing file or differing columns.
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


def _relative_difference(left: str, right: str) -> float | None:
    """Return the relative difference of two cells, or None if they are not both numeric."""
    try:
        a, b = float(left), float(right)
    except ValueError:
        return None
    return abs(a - b) / max(abs(a), abs(b), _SCALE_FLOOR)


def compare_output_dirs(
    expected: Path,
    actual: Path,
    rtol: float = DEFAULT_RTOL,
    ignore: Collection[str] = (),
) -> ComparisonResult:
    """Compare the ``tables/`` CSVs of two output directories.

    Args:
        expected: Reference output directory.
        actual: Directory to check.
        rtol: Relative tolerance treated as agreement.
        ignore: File names to leave out of the comparison. Use this only for tables that are
            passthroughs of the raw inputs, whose length depends on the vintage of the
            downloaded data rather than on anything the study computes.

    Returns:
        The comparison outcome.

    Raises:
        FileNotFoundError: If the reference directory has no ``tables/`` subdirectory.
    """
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
                relative = _relative_difference(reference_value, candidate_value)
                if relative is None:
                    discrepancies.append(
                        f"{reference_file.name} line {number} [{column}]: "
                        f"{reference_value!r} expected, {candidate_value!r} found"
                    )
                    continue
                if relative > worst:
                    worst, worst_field = relative, f"{reference_file.name}:{column}"
                if relative > rtol:
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
