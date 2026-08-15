#!/usr/bin/env python3
"""Assert that a fresh pipeline run reproduces the committed results.

Run ``python run_pipeline.py`` first, then this script.

Why this is a numerical comparison and not a byte comparison
-----------------------------------------------------------
Within one environment this pipeline is bit-reproducible: it is seeded, single
threaded in its own logic, and writes no timestamps into its CSV output. Across
environments it is not, and that is expected rather than a defect. Regression
coefficients, matrix inversions and eigenvalue-based diagnostics are computed through
BLAS/LAPACK, whose reduction order depends on the library build, its threading
decisions and the CPU's available instruction sets. Two correct implementations can
therefore differ in the last bits of a float, and once a value is written at full
precision that difference shows up as a changed line of text.

So the check that matters is: does every number agree to within numerical tolerance,
and is every artifact present? The maximum observed difference is printed, so drift
that is genuinely too large to be rounding shows up in the log rather than being
absorbed by the tolerance.

PNG and XLSX files are checked for presence rather than content. Matplotlib embeds
renderer-version metadata and openpyxl embeds a creation timestamp, so neither is
byte-stable across environments by design.
"""

from __future__ import annotations

import io
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

# numpy.isclose convention: |a - b| <= ATOL + RTOL * |b|
RTOL = 1e-6
ATOL = 1e-10

EXPECTED_CHARTS = 17
EXPECTED_TABLES = 20
EXPECTED_DOCS = ("RESEARCH_REPORT.md", "INTERVIEW_SHEET.md", "DATA_SOURCES.md")


def committed(path: str) -> pd.DataFrame | None:
    """Read a CSV as it exists in HEAD, or None if it is not tracked."""
    result = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, cwd=config.ROOT)
    if result.returncode != 0:
        return None
    return pd.read_csv(io.BytesIO(result.stdout))


def compare(name: str, expected: pd.DataFrame, actual: pd.DataFrame) -> tuple[bool, float, list[str]]:
    problems: list[str] = []
    if list(expected.columns) != list(actual.columns):
        problems.append(f"columns changed: {set(expected.columns) ^ set(actual.columns)}")
        return False, float("nan"), problems
    if len(expected) != len(actual):
        problems.append(f"row count changed: {len(expected)} -> {len(actual)}")
        return False, float("nan"), problems

    worst = 0.0
    for column in expected.columns:
        left, right = expected[column], actual[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            a, b = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                problems.append(f"{column}: NaN pattern changed")
                continue
            # Infinities are compared by position and sign rather than by subtraction.
            # They are meaningful values here, not artifacts: a rank-deficient regression
            # reports an infinite condition number, and that must stay infinite.
            # np.where rather than multiplication, so a NaN entry yields 0 instead of
            # propagating NaN and making array_equal false for any column with gaps.
            if not np.array_equal(
                np.where(np.isinf(a), np.sign(a), 0.0), np.where(np.isinf(b), np.sign(b), 0.0)
            ):
                problems.append(f"{column}: infinity pattern changed")
                continue
            # Non-finite entries are excluded before subtracting, not after, so the
            # comparison cannot emit invalid-value warnings.
            finite = np.isfinite(b)
            difference = np.abs(a[finite] - b[finite])
            allowed = ATOL + RTOL * np.abs(b[finite])
            if difference.size:
                worst = max(worst, float(np.nanmax(difference)))
                exceeded = int((difference > allowed).sum())
                if exceeded:
                    index = int(np.nanargmax(difference - allowed))
                    problems.append(
                        f"{column}: {exceeded} value(s) beyond tolerance, worst "
                        f"{b[finite][index]:.10g} -> {a[finite][index]:.10g}"
                    )
        elif not left.astype(str).equals(right.astype(str)):
            changed = int((left.astype(str) != right.astype(str)).sum())
            problems.append(f"{column}: {changed} non-numeric value(s) changed")
    return not problems, worst, problems


def main() -> int:
    failures: list[str] = []
    worst_overall, worst_source = 0.0, ""

    charts = sorted(config.CHART_DIR.glob("*.png"))
    if len(charts) != EXPECTED_CHARTS:
        failures.append(f"expected {EXPECTED_CHARTS} charts, found {len(charts)}")
    empty = [c.name for c in charts if c.stat().st_size == 0]
    if empty:
        failures.append(f"empty chart files: {empty}")

    if not config.EXCEL_PATH.exists() or config.EXCEL_PATH.stat().st_size == 0:
        failures.append("Excel monitor missing or empty")

    for document in EXPECTED_DOCS:
        path = config.DOCS_DIR / document
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"document missing or empty: {document}")

    tables = sorted(config.TABLE_DIR.glob("*.csv"))
    if len(tables) != EXPECTED_TABLES:
        failures.append(f"expected {EXPECTED_TABLES} tables, found {len(tables)}")

    for table in tables:
        relative = table.relative_to(config.ROOT).as_posix()
        expected = committed(relative)
        if expected is None:
            print(f"  skip     {table.name} (not tracked in HEAD)")
            continue
        ok, worst, problems = compare(table.name, expected, pd.read_csv(table))
        if worst > worst_overall:
            worst_overall, worst_source = worst, table.name
        if ok:
            print(f"  match    {table.name:<34} max abs diff {worst:.3e}")
        else:
            print(f"  MISMATCH {table.name}")
            failures.extend(f"{table.name}: {p}" for p in problems)

    print()
    print(f"tolerance: |a - b| <= {ATOL:g} + {RTOL:g} * |b|")
    print(f"largest difference anywhere: {worst_overall:.3e} (in {worst_source or 'n/a'})")
    print(f"artifacts: {len(charts)} charts, {len(tables)} tables, {len(EXPECTED_DOCS)} documents, 1 workbook")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll numeric results reproduce the committed values within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
