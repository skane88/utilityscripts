"""
File to test the math utilities file
"""

from math import isclose

import pytest

from utilityscripts.math_utils import m_ceil, m_floor, m_round, round_significant


@pytest.mark.parametrize(
    "x, base,expected",
    [
        (100, 1, 100),
        (-100, 1, -100),
        (12.34, 0.5, 12.5),
        (-12.34, 0.5, -12.5),
        (12.34, -0.5, 12.5),
        (-12.34, -0.5, -12.5),
        (1.30, 0.05, 1.30),
        (0.1 * 3 * 10, 1, 3),
        (4.6, 0.2, 4.6),
        (1.39, 0.05, 1.40),
        (4.3, 0.25, 4.25),
    ],
)
def test_m_round(x, base, expected):
    assert m_round(x, base) == expected


@pytest.mark.parametrize(
    "x, base, float_tolerance, expected",
    [
        (100, 1, None, 100),
        (-100, 1, None, -100),
        (12.34, 0.5, None, 12.0),
        (-12.34, 0.5, None, -12.5),
        (12.34, -0.5, None, 12.0),
        (-12.34, -0.5, None, -12.5),
        (1.30, 0.05, None, 1.30),
        (0.1 * 3 * 10, 1, None, 3),
        (4.6, 0.2, None, 4.6),
        (1.39, 0.05, None, 1.35),
        (4.3, 0.25, None, 4.25),
        (0.119999999998, 0.001, None, 0.119),
        (0.119999999998, 0.001, 1e-6, 0.120),
    ],
)
def test_m_floor(x, base, float_tolerance, expected):
    assert m_floor(x, base, float_tolerance=float_tolerance) == expected


@pytest.mark.parametrize(
    "x, base, float_tolerance, expected",
    [
        (100, 1, None, 100),
        (-100, 1, None, -100),
        (12.34, 0.5, None, 12.5),
        (-12.34, 0.5, None, -12.0),
        (12.34, -0.5, None, 12.5),
        (-12.34, -0.5, None, -12.0),
        (1.30, 0.05, None, 1.30),
        (
            0.1 * 3 * 10,
            1,
            None,
            4.0,
        ),  # this is a tricky one - 0.1*3/10 != 3.0 in floating point.
        (4.6, 0.2, None, 4.6),
        (1.39, 0.05, None, 1.40),
        (4.3, 0.25, None, 4.50),
        (0.1200000001, 0.001, None, 0.121),
        (0.1200000001, 0.001, 1e-6, 0.120),
    ],
)
def test_m_ceil(x, base, float_tolerance, expected):
    assert m_ceil(x, base, float_tolerance=float_tolerance) == expected


@pytest.mark.parametrize(
    "x, s, expected",
    [
        (123.456, 2, 120),
        (123.456, 3, 123),
        (123.456, 4, 123.5),
        (-123.456, 2, -120),
        (-123.456, 3, -123),
        (-123.456, 4, -123.5),
        (0.123456, 1, 0.1),
        (0.123456, 2, 0.12),
        (0.123456, 3, 0.123),
        (0.123456, 4, 0.1235),
        (-0.123456, 1, -0.1),
        (-0.123456, 2, -0.12),
        (-0.123456, 3, -0.123),
        (-0.123456, 4, -0.1235),
    ],
)
def test_round_significant(x, s, expected):
    assert isclose(round_significant(x, s), expected, abs_tol=1e-9)
