"""Tests for the reproducibility comparison.

This function decides whether a CI run is allowed to pass, so its NaN and infinity
handling has to be right. Two bugs were found in it during development: subtracting
infinities emitted spurious warnings, and encoding the infinity pattern by multiplication
made any column containing a NaN compare unequal to itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_reproducibility import ATOL, RTOL, compare  # noqa: E402


def _frame(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_identical_frames_match():
    frame = _frame(a=[1.0, 2.0, 3.0], label=["x", "y", "z"])
    ok, worst, problems = compare("t", frame, frame.copy())
    assert ok and worst == 0.0 and not problems


def test_float_noise_within_tolerance_matches():
    expected = _frame(a=[1.0, 2.0, 3.0])
    actual = _frame(a=[1.0 + 1e-13, 2.0 - 2e-13, 3.0 + 3e-13])
    ok, worst, problems = compare("t", expected, actual)
    assert ok, problems
    assert 0 < worst < 1e-11


def test_material_change_fails():
    """A 0.1% change is invisible in rounded output and must still be caught."""
    expected = _frame(a=[1.0, 2.0, 3.0])
    actual = _frame(a=[1.0, 2.0 * 1.001, 3.0])
    ok, _, problems = compare("t", expected, actual)
    assert not ok
    assert "beyond tolerance" in problems[0]


def test_tolerance_boundary():
    expected = _frame(a=[1.0])
    just_inside = _frame(a=[1.0 + 0.5 * (ATOL + RTOL)])
    just_outside = _frame(a=[1.0 + 10.0 * (ATOL + RTOL)])
    assert compare("t", expected, just_inside)[0]
    assert not compare("t", expected, just_outside)[0]


def test_matching_nan_pattern_matches():
    frame = _frame(a=[1.0, np.nan, 3.0])
    ok, _, problems = compare("t", frame, frame.copy())
    assert ok, problems


def test_changed_nan_pattern_fails():
    expected = _frame(a=[1.0, np.nan, 3.0])
    actual = _frame(a=[1.0, 2.0, 3.0])
    ok, _, problems = compare("t", expected, actual)
    assert not ok
    assert "NaN pattern changed" in problems[0]


def test_preserved_infinity_matches():
    """A rank-deficient regression reports an infinite condition number by design."""
    frame = _frame(condition_number=[np.inf, 70.8, 73.7])
    ok, _, problems = compare("t", frame, frame.copy())
    assert ok, problems


def test_infinity_replaced_by_a_finite_value_fails():
    """Regressing to reporting a floating-point artifact must fail the check."""
    expected = _frame(condition_number=[np.inf, 70.8])
    actual = _frame(condition_number=[1.9e15, 70.8])
    ok, _, problems = compare("t", expected, actual)
    assert not ok
    assert "infinity pattern changed" in problems[0]


def test_nan_and_infinity_together_do_not_false_positive():
    frame = _frame(a=[np.inf, np.nan, 1.0, -np.inf])
    ok, _, problems = compare("t", frame, frame.copy())
    assert ok, problems


def test_sign_of_infinity_matters():
    expected = _frame(a=[np.inf])
    actual = _frame(a=[-np.inf])
    assert not compare("t", expected, actual)[0]


def test_changed_text_column_fails():
    expected = _frame(a=[1.0], result=["PASS"])
    actual = _frame(a=[1.0], result=["FAIL"])
    ok, _, problems = compare("t", expected, actual)
    assert not ok
    assert "non-numeric" in problems[0]


def test_changed_shape_fails():
    expected = _frame(a=[1.0, 2.0])
    assert not compare("t", expected, _frame(a=[1.0]))[0]
    assert not compare("t", expected, _frame(b=[1.0, 2.0]))[0]
