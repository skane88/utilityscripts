"""
File to test the math utilities file
"""

import sys
from decimal import Decimal
from math import isclose

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from utilityscripts.math_utils import (
    engineering_number,
    m_ceil,
    m_floor,
    m_round,
    round_significant,
    scientific_number,
)


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
        (0, 1, 0),
    ],
)
def test_m_round(x, base, expected):
    assert m_round(x, base) == expected


@given(
    st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.decimals(allow_nan=False, allow_infinity=False)
    | st.fractions(),
    st.integers().filter(lambda n: n != 0),
)
@settings(max_examples=1000)
def test_m_round_hypothesis(x, base):
    """
    Test m_round with hypothesis.

    No checks for accuracy but will rule out unusual / spurious input.
    """

    m_round(x, base)


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


@given(
    st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.decimals(allow_nan=False, allow_infinity=False)
    | st.fractions(),
    st.integers().filter(lambda n: n != 0),
)
@settings(max_examples=1000)
def test_m_floor_hypothesis(x, base):
    """
    Test m_round with hypothesis.

    No checks for accuracy but will rule out unusual / spurious input.
    """

    m_floor(x, base)


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


@given(
    st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.decimals(allow_nan=False, allow_infinity=False)
    | st.fractions(),
    st.integers().filter(lambda n: n != 0),
)
@settings(max_examples=1000)
def test_m_ceil_hypothesis(x, base):
    """
    Test m_round with hypothesis.

    No checks for accuracy but will rule out unusual / spurious input.
    """

    m_ceil(x, base)


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
        (0, 3, 0.0),
        (1234.567, 1, 1000.0),
        (1234.567, 3, 1230.0),
        (1234.567, 5, 1234.6),
        (-12345.678, 1, -10000.0),
        (-12345.678, 4, -12350.0),
        (-12345.678, 6, -12345.7),
        (1.234567859e99, 5, 1.2346e99),
        (-1.234567859e99, 5, -1.2346e99),
        (1.23456789e-99, 5, 1.2346e-99),
        (-1.23456789e-99, 5, 1.2346e-99),
    ],
)
def test_round_significant(x, s, expected):
    assert isclose(round_significant(x, s), expected, abs_tol=1e-9)


@pytest.mark.parametrize(
    "value, expected",
    [
        (12.3456789, (1.23456789, 1)),
        (-12.3456789, (-1.23456789, 1)),
        (-1000000, (-1, 6)),
        (-100000, (-1, 5)),
        (-10000, (-1, 4)),
        (-1000, (-1, 3)),
        (-100, (-1, 2)),
        (-10, (-1, 1)),
        (-1, (-1, 0)),
        (0, (0, 0)),
        (1, (1, 0)),
        (10, (1, 1)),
        (100, (1, 2)),
        (1000, (1, 3)),
        (10000, (1, 4)),
        (100000, (1, 5)),
        (1000000, (1, 6)),
        (123456, (1.23456, 5)),
        (12345.6, (1.23456, 4)),
        (1234.56, (1.23456, 3)),
        (123.456, (1.23456, 2)),
        (12.3456, (1.23456, 1)),
        (1.23456, (1.23456, 0)),
        (0.123456, (1.23456, -1)),
        (0.0123456, (1.23456, -2)),
        (0.00123456, (1.23456, -3)),
        (0.000123456, (1.23456, -4)),
        (0.0000123456, (1.23456, -5)),
        (0.00000123456, (1.23456, -6)),
        (0.000000123456, (1.23456, -7)),
        (0.0000000123456, (1.23456, -8)),
        (0.00000000123456, (1.23456, -9)),
        (0.000000000123456, (1.23456, -10)),
        (-0.000000123456, (-1.23456, -7)),
        (1e-5, (1, -5)),  # test failing in hypothesis.
    ],
)
def test_sci_num(value, expected):
    result = scientific_number(value)

    assert isclose(result[0], expected[0], rel_tol=1e-9)
    assert result[1] == expected[1]


@given(
    st.integers()
    | st.floats(allow_nan=False, allow_infinity=False).filter(
        lambda n: abs(n) > sys.float_info.min
    )
    | st.decimals(allow_nan=False, allow_infinity=False)
    | st.fractions()
)
@settings(max_examples=1000)
def test_sci_num_hypothesis(value):
    """
    Test the sci_not function with hypothesis.
    """

    result = scientific_number(value)

    assert isclose(value, result[0] * Decimal("10") ** result[1], rel_tol=1e-9)
    assert (1 <= abs(result[0]) < 10) or (result[0] == 0)  # noqa: PLR2004


@pytest.mark.parametrize(
    "value, expected",
    [
        (12.3456789, (12.3456789, 0)),
        (-12.3456789, (-12.3456789, 0)),
        (-1000000000, (-1, 9)),
        (-100000000, (-100, 6)),
        (-10000000, (-10, 6)),
        (-1000000, (-1, 6)),
        (-100000, (-100, 3)),
        (-10000, (-10, 3)),
        (-1000, (-1, 3)),
        (-100, (-100, 0)),
        (-10, (-10, 0)),
        (-1, (-1, 0)),
        (0, (0, 0)),
        (1, (1, 0)),
        (10, (10, 0)),
        (100, (100, 0)),
        (1000, (1, 3)),
        (10000, (10, 3)),
        (100000, (100, 3)),
        (1000000, (1, 6)),
        (10000000, (10, 6)),
        (100000000, (100, 6)),
        (1000000000, (1, 9)),
        (0.123456, (123.456, -3)),
        (0.0123456, (12.3456, -3)),
        (0.00123456, (1.23456, -3)),
        (0.000123456, (123.456, -6)),
        (0.0000123456, (12.3456, -6)),
        (0.00000123456, (1.23456, -6)),
        (0.000000123456, (123.456, -9)),
        (0.0000000123456, (12.3456, -9)),
        (0.00000000123456, (1.23456, -9)),
        (0.000000000123456, (123.456, -12)),
        (-0.000000123456, (-123.456, -9)),
    ],
)
def test_engineering_number(value, expected):
    result = engineering_number(value)

    assert isclose(result[0], expected[0], rel_tol=1e-9)
    assert result[1] == expected[1]
    assert result[1] % 3 == 0
    assert (1 <= abs(result[0]) < 1000) or result[0] == 0  # noqa: PLR2004


@given(
    st.integers()
    | st.floats(allow_nan=False, allow_infinity=False).filter(
        lambda n: abs(n) > sys.float_info.min
    )
    | st.decimals(allow_nan=False, allow_infinity=False)
    | st.fractions()
)
@settings(max_examples=1000)
def test_eng_num_hypothesis(value):
    """
    Test the engineering_number function with hypothesis.
    """

    result = engineering_number(value)

    assert isclose(value, result[0] * Decimal("10") ** result[1], rel_tol=1e-9)
    assert result[1] % 3 == 0
    assert (1 <= abs(result[0]) < 1000) or result[0] == 0  # noqa: PLR2004
