"""
Test the Result class.
"""

import pytest

from utilityscripts.result import ResultError, Variable


def test_variable():
    """
    Minimal test of the Variable class.
    """

    # Test default values
    v = Variable(1)
    assert v.value == 1
    assert v.symbol is None
    assert v.units is None
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
    assert v.str_value == f"{value:.2f} {units}"


@pytest.mark.parametrize(
    "val, expected",
    [
        (Variable(2), "None=2.000e+00"),
        (Variable(1, units="m"), "None=1.000e+00 m"),
        (Variable(1, units="m", fmt_string=".2f"), "None=1.00 m"),
        (Variable(1, units="m", fmt_string=".2e"), "None=1.00e+00 m"),
        (Variable(2, symbol="a"), "a=2.000e+00"),
        (Variable(1, symbol="a", units="m"), "a=1.000e+00 m"),
        (Variable(1, symbol="a", units="m", fmt_string=".2f"), "a=1.00 m"),
        (Variable(1, symbol="a", units="m", fmt_string=".2e"), "a=1.00e+00 m"),
    ],
)
def test_variable_string(val, expected):
    assert str(val) == expected


def test_result_error():
    with pytest.raises(ResultError):
        Variable(2.0, units="m", fmt_string=".2%")


@pytest.mark.parametrize(
    "val, expected",
    [
        (Variable(1), "1.000 \\times 10^{0}"),
        (Variable(1, fmt_string=".3E"), "1.000 \\times 10^{0}"),
        (Variable(1, units="m"), "1.000 \\times 10^{0} \\text{m}"),
        (Variable(1, symbol="a"), "\\text{a} = 1.000 \\times 10^{0}"),
        (
            Variable(1, symbol="a", units="m", fmt_string=".2f"),
            "\\text{a} = 1.00 \\text{m}",
        ),
        (
            Variable(2.12345, symbol="a", units="m", fmt_string=".5g"),
            "\\text{a} = 2.1235 \\text{m}",
        ),
        (
            Variable(0.0012, symbol="a", units="m", fmt_string=".2g"),
            "\\text{a} = 0.0012 \\text{m}",
        ),
        (
            Variable(0.000012, symbol="a", units="m", fmt_string=".2g"),
            "\\text{a} = 1.2 \\times 10^{-5} \\text{m}",
        ),
        (
            Variable(214, symbol="a", units="m", fmt_string=".4g"),
            "\\text{a} = 214 \\text{m}",
        ),
        (
            Variable(2145, symbol="a", units="m", fmt_string=".3g"),
            "\\text{a} = 2.14 \\times 10^{3} \\text{m}",
        ),
        (Variable(0.123, symbol="a", fmt_string=".3%"), "\\text{a} = 12.300\\%"),
        (Variable(0.00123, symbol="a", fmt_string=".3%"), "\\text{a} = 0.123\\%"),
    ],
)
def test_latex_string(val, expected):
    assert val.latex_string == expected
