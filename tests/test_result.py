"""
Test the Result class.
"""

import pytest

from utilityscripts.result import DeprecatedResult, Variable, _sci_not


def test_variable():
    """
    Minimal test of the Variable class.
    """

    # Test default values
    v = Variable(1)
    assert v.value == 1
    assert v.symbol is None
    assert v.units == ""
    assert v.fmt_string == ".3e"
    assert v.str_value == "1.000e+00"

    value = 2
    symbol = "b"
    units = "m"
    fmt_string = ".2f"

    v = Variable(value, symbol=symbol, units=units, fmt_string=fmt_string)
    assert v.value == value
    assert v.symbol == symbol
    assert v.units == units
    assert v.fmt_string == fmt_string
    assert v.str_value == f"{value:.2f}{units}"


@pytest.mark.parametrize("operation", [str, float, int, bool, complex, bytes])
def test_result_to_str(operation):
    r = DeprecatedResult(1)
    assert operation(r) == operation(1)


def test_result_neg():
    r = DeprecatedResult(1)
    assert -r == -1


def test_result_pos():
    r = DeprecatedResult(1)
    assert +r == +1


def test_result_invert():
    r = DeprecatedResult(1)
    assert ~r == ~1


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 1 + 2),
        (2, 1, 2 + 1),
        ("a", "b", "a" + "b"),
        ("b", "a", "b" + "a"),
        (True, True, 2),
        (True, False, 1),
        (False, False, 0),
    ],
)
def test_result_add(result, other, expected):
    result = DeprecatedResult(result)
    assert result + other == expected


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 1 - 2),
        (2, 1, 2 - 1),
        (1.0, 2.0, 1.0 - 2.0),
        (2.0, 1.0, 2.0 - 1.0),
        (True, True, 0),
        (True, False, 1),
        (False, False, 0),
    ],
)
def test_result_sub(result, other, expected):
    result = DeprecatedResult(result)
    assert result - other == expected


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 1 * 2),
        (1, 0.5, 1 * 0.5),
        (1, -1, 1 * -1),
        (1, -0.5, 1 * -0.5),
        (1, 0, 1 * 0),
        ("a", 2, "a" * 2),
        (2, "a", 2 * "a"),
    ],
)
def test_result_mul(result, other, expected):
    result = DeprecatedResult(result)
    assert result * other == expected
    assert other * result == expected


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 1 / 2),
        (1, 0.5, 1 / 0.5),
    ],
)
def test_result_truediv(result, other, expected):
    result = DeprecatedResult(result)
    assert result / other == expected


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 2 / 1),
        (1, 0.5, 0.5 / 1),
    ],
)
def test_result_rtruediv(result, other, expected):
    result = DeprecatedResult(result)
    assert other / result == expected


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 1 // 2),
        (1, 0.5, 1 // 0.5),
    ],
)
def test_result_floordiv(result, other, expected):
    result = DeprecatedResult(result)
    assert result // other == expected


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 2 // 1),
        (1, 0.5, 0.5 // 1),
    ],
)
def test_result_rfloordiv(result, other, expected):
    result = DeprecatedResult(result)
    assert other // result == expected


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 1**2),
        (1, 0.5, 1**0.5),
    ],
)
def test_result_pow(result, other, expected):
    result = DeprecatedResult(result)
    assert result**other == expected


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 2**1),
        (1, 0.5, 0.5**1),
    ],
)
def test_result_rpow(result, other, expected):
    result = DeprecatedResult(result)
    assert other**result == expected


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
    ],
)
def test_sci_not(value, expected):
    result = _sci_not(value)

    assert result[0] == expected[0]
    assert result[1] == expected[1]
