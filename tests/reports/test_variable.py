"""
Test the Result class.
"""

import pytest

from utilityscripts.reports.report import ResultError, Variable


def test_variable():
    """
    Minimal test of the Variable class.
    """

    # Test default values
    v = Variable(1)
    assert v.value == 1
    assert v.symbol is None
    assert v.units is None
    assert v.fmt_string is None

    value = 2
    symbol = "b"
    units = "m"
    fmt_string = ".2f"

    v = Variable(value, symbol=symbol, units=units, fmt_string=fmt_string)
    assert v.value == value
    assert v.symbol == symbol
    assert v.units == units
    assert v.fmt_string == fmt_string


@pytest.mark.parametrize(
    "val, expected",
    [
        (Variable(2), "2"),
        (Variable(1, units="m"), "1m"),
        (Variable(1, units="m", fmt_string=".2f"), "1.00m"),
        (Variable(1, units="m", fmt_string=".2e"), "1.00e+00m"),
        (Variable(2, symbol="a"), "a=2"),
        (Variable(1, symbol="a", units="m"), "a=1m"),
        (Variable(1, symbol="a", units="m", fmt_string=".2f"), "a=1.00m"),
        (Variable(1, symbol="a", units="m", fmt_string=".2e"), "a=1.00e+00m"),
        (Variable(2, fmt_string=None), "2"),
        (Variable("abc"), "abc"),
        (Variable("abc", units="m"), "abc m"),
        (Variable(None, symbol="a"), "a=None"),
        (Variable([]), "[]"),
        (Variable(set()), "{}"),
        (Variable({}), "{}"),
        (Variable([1, 2, 3]), "[1, 2, 3]"),
        (Variable({"a": 1, "b": 2, "c": 3}), "{'a': 1, 'b': 2, 'c': 3}"),
        (Variable({1, 2, 3}), "{1, 2, 3}"),
        (Variable(list(range(0, 6))), "[0, 1, 2, 3, 4, 5]"),
        (
            Variable({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}),
            "{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}",
        ),
        (Variable(list(range(0, 7))), "[0, 1, 2, 3, 4, ..., 6]"),
        (
            Variable({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}),
            "{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}",
        ),
        (Variable(list(range(0, 100))), "[0, 1, 2, 3, 4, ..., 99]"),
        (
            Variable({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}, shorten_list=4),
            "{'a': 1, 'b': 2, 'c': 3, ..., 'f': 6}",
        ),
        (
            Variable({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}, shorten_list=4),
            "{'a': 1, 'b': 2, 'c': 3, ..., 'e': 5}",
        ),
        (
            Variable({"a": 1, "b": 2, "c": 3, "d": 4}, shorten_list=4),
            "{'a': 1, 'b': 2, 'c': 3, 'd': 4}",
        ),
        (Variable(set(range(0, 100))), "{0, 1, 2, 3, 4, ..., 99}"),
        (Variable(set(range(0, 100)), shorten_list=3), "{0, 1, ..., 99}"),
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
        (Variable(1), "$1$"),
        (Variable(1, fmt_string=".3E"), "$1.000 \\times 10^{0}$"),
        (Variable(1, units="m"), "$1\\text{m}$"),
        (Variable(1, symbol="a"), "$\\text{a} = 1$"),
        (
            Variable(1, symbol="a", units="m", fmt_string=".2f"),
            "$\\text{a} = 1.00\\text{m}$",
        ),
        (
            Variable(2.12345, symbol="a", units="m", fmt_string=".5g"),
            "$\\text{a} = 2.1235\\text{m}$",
        ),
        (
            Variable(0.0012, symbol="a", units="m", fmt_string=".2g"),
            "$\\text{a} = 0.0012\\text{m}$",
        ),
        (
            Variable(0.000012, symbol="a", units="m", fmt_string=".2g"),
            "$\\text{a} = 1.2 \\times 10^{-5}\\text{m}$",
        ),
        (
            Variable(214, symbol="a", units="m", fmt_string=".4g"),
            "$\\text{a} = 214\\text{m}$",
        ),
        (
            Variable(2145, symbol="a", units="m", fmt_string=".3g"),
            "$\\text{a} = 2.14 \\times 10^{3}\\text{m}$",
        ),
        (Variable(0.123, symbol="a", fmt_string=".3%"), "$\\text{a} = 12.300\\%$"),
        (Variable(0.00123, symbol="a", fmt_string=".3%"), "$\\text{a} = 0.123\\%$"),
        (Variable("abc", symbol="a"), "$\\text{a} = \\text{abc}$"),
        (Variable("I was here!", symbol="a"), "$\\text{a} = \\text{I was here!}$"),
        (Variable("abc", units="m"), "$\\text{abc}\\ \\text{m}$"),
        (Variable(None, symbol="a"), "$\\text{a} = \\text{None}$"),
        (Variable([]), "$\\left[\\right]$"),
        (Variable(set()), "$\\left{\\right}$"),
        (Variable({}), "$\\left{\\right}$"),
        (Variable([1, 2, 3]), "$\\left[1, 2, 3\\right]$"),
        (Variable(list(range(0, 6))), "$\\left[0, 1, 2, 3, 4, 5\\right]$"),
        (
            Variable({"a": 1, "b": 2, "c": 3}),
            "$\\left{\\text{a}: 1, \\text{b}: 2, \\text{c}: 3\\right}$",
        ),
        (
            Variable({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}),
            "$\\left{\\text{a}: 1, \\text{b}: 2, \\text{c}: 3, \\text{d}: 4, \\text{e}: 5, \\text{f}: 6\\right}$",
        ),
        (Variable({1, 2, 3}), "$\\left{1, 2, 3\\right}$"),
        (Variable(list(range(0, 100))), "$\\left[0, 1, 2, 3, 4, ..., 99\\right]$"),
        (
            Variable({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}, shorten_list=4),
            "$\\left{\\text{a}: 1, \\text{b}: 2, \\text{c}: 3, ..., \\text{f}: 6\\right}$",
        ),
        (Variable(set(range(0, 100))), "$\\left{0, 1, 2, 3, 4, ..., 99\\right}$"),
    ],
)
def test_latex_string(val, expected):
    assert val.latex_string == expected


def test_latex_disabled():
    val = Variable(1, units="m", fmt_string=".2f")
    assert val.latex_string == "$1.00\\text{m}$"
    assert "text/latex" in val._repr_mimebundle_()

    val = Variable(1, units="m", fmt_string=".2f", disable_latex=True)

    assert val.latex_string is None
    assert "text/latex" not in val._repr_mimebundle_()
