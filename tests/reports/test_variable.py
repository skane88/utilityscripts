"""
Test the Result class.
"""

import pytest
import sympy as sym
from hypothesis import given
from hypothesis import strategies as st

from utilityscripts.reports.report import ResultError, Variable, format_sig_figs


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

    assert repr(v)

    value = 2
    symbol = "b"
    units = "m"
    fmt_string = ".2f"

    v = Variable(
        value, symbol=symbol, units=units, fmt_string=fmt_string, use_repr_latex=False
    )
    assert v.value == value
    assert v.symbol == symbol
    assert v.units == units
    assert v.fmt_string == fmt_string
    assert v.use_repr_latex is False


@pytest.mark.parametrize(
    "val, expected",
    [
        (Variable(2), "2"),
        (Variable(1, units="m"), "1m"),
        (Variable(1, units="m", fmt_string=".2f"), "1.00m"),
        (Variable(1, units="m", fmt_string=".2e"), "1.00e+00m"),
        (Variable(1.2345, units="m", fmt_string=".2j"), "1.2m"),
        (Variable(123456, units="m", fmt_string=".2j"), "120e+03m"),
        (Variable(2, symbol="a"), "a=2"),
        (Variable(1, symbol="a", units="m"), "a=1m"),
        (Variable(1, symbol="a", units="m", fmt_string=".2f"), "a=1.00m"),
        (Variable(1, symbol="a", units="m", fmt_string=".2e"), "a=1.00e+00m"),
        (Variable(2, fmt_string=None), "2"),
        (Variable("abc"), "'abc'"),
        (Variable("abc", units="m"), "'abc' m"),
        (Variable(None, symbol="a"), "a=None"),
        (Variable([]), "[]"),
        (Variable(set()), "{}"),
        (Variable({}), "{}"),
        (Variable([1, 2, 3]), "[1, 2, 3]"),
        (Variable({"a": 1, "b": 2, "c": 3}), "{'a': 1, 'b': 2, 'c': 3}"),
        (Variable({1, 2, 3}), "{1, 2, 3}"),
        (Variable(["a", "b", "c"]), "['a', 'b', 'c']"),
        (Variable(list(range(0, 6))), "[0, 1, 2, 3, 4, 5]"),
        (
            Variable({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}),
            "{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}",
        ),
        (Variable(list(range(0, 7))), "[0, 1, 2, 3, 4, ..., 6]"),
        (Variable(list(range(0, 7)), shorten_list=None), "[0, 1, 2, 3, 4, 5, 6]"),
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
            Variable({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8}),
            "{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, ..., 'h': 8}",
        ),
        (
            Variable(
                {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8},
                shorten_list=None,
            ),
            "{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}",
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
        (Variable("alpha"), "'α'"),  # noqa: RUF001
        (Variable([["a", "b", "c"], "d"]), "[['a', 'b', 'c'], 'd']"),
        (Variable([[["a", "b"], "c"], "d"]), "[[[...], 'c'], 'd']"),
        (
            Variable([[[["a"], "b"], "c"], "d"], max_depth=5),
            "[[[['a'], 'b'], 'c'], 'd']",
        ),
        (
            Variable({"a": {"b": {"c": {"d": 4}}}}),
            "{'a': {'b': {...}}}",
        ),
        (
            Variable({"a": {"b": {"c": {"d": 4}}}}, max_depth=5),
            "{'a': {'b': {'c': {'d': 4}}}}",
        ),
        (Variable({"a": 1, "b": {"c": 3}}), "{'a': 1, 'b': {'c': 3}}"),
        (Variable({"a": 1, "b": {"c": {"d": 4}}}, max_depth=1), "{'a': 1, 'b': {...}}"),
        (Variable(2, symbol=("a", "\\text{a}")), "a=2"),
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
        (Variable(1.2345e6, fmt_string=".2J"), "$1.2 \\times 10^{6}$"),
        (Variable(1.2345, fmt_string=".4J"), "$1.234$"),
        # note: tricky one, as 1.2345 in floating point is actually 1.2344999999, which
        # is ever so slightly less than 1.2345.
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
        (Variable(set()), "$\\left\\{\\right\\}$"),
        (Variable({}), "$\\left\\{\\right\\}$"),
        (Variable([1, 2, 3]), "$\\left[1, 2, 3\\right]$"),
        (Variable(list(range(0, 6))), "$\\left[0, 1, 2, 3, 4, 5\\right]$"),
        (Variable(list(range(0, 7))), "$\\left[0, 1, 2, 3, 4, ..., 6\\right]$"),
        (
            Variable(list(range(0, 7)), shorten_list=None),
            "$\\left[0, 1, 2, 3, 4, 5, 6\\right]$",
        ),
        (
            Variable({"a": 1, "b": 2, "c": 3}),
            "$\\left\\{\\text{a}: 1, \\text{b}: 2, \\text{c}: 3\\right\\}$",
        ),
        (
            Variable({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}),
            "$\\left\\{\\text{a}: 1, \\text{b}: 2, \\text{c}: 3, \\text{d}: 4, \\text{e}: 5, \\text{f}: 6\\right\\}$",
        ),
        (
            Variable({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7}),
            "$\\left\\{\\text{a}: 1, \\text{b}: 2, \\text{c}: 3, \\text{d}: 4, \\text{e}: 5, ..., \\text{g}: 7\\right\\}$",
        ),
        (
            Variable(
                {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7},
                shorten_list=None,
            ),
            "$\\left\\{\\text{a}: 1, \\text{b}: 2, \\text{c}: 3, \\text{d}: 4, \\text{e}: 5, \\text{f}: 6, \\text{g}: 7\\right\\}$",
        ),
        (Variable({1, 2, 3}), "$\\left\\{1, 2, 3\\right\\}$"),
        (Variable(list(range(0, 100))), "$\\left[0, 1, 2, 3, 4, ..., 99\\right]$"),
        (
            Variable({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}, shorten_list=4),
            "$\\left\\{\\text{a}: 1, \\text{b}: 2, \\text{c}: 3, ..., \\text{f}: 6\\right\\}$",
        ),
        (Variable(set(range(0, 100))), "$\\left\\{0, 1, 2, 3, 4, ..., 99\\right\\}$"),
        (Variable("alpha"), "$\\alpha$"),
        (Variable("Rho"), "$Ρ$"),  # noqa: RUF001
        (Variable(2, symbol=("a", "\\phi N")), "$\\phi N = 2$"),
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


def test_repr_latex():
    """
    Test that the latex_string method matches an existing _repr_latex_ method.
    """

    a = sym.symbols("a")
    v = Variable(a)

    assert v.latex_string == a._repr_latex_()

    a, b, c = sym.symbols("a b c")
    eqn = a + b / c
    v = Variable(eqn)

    assert v.latex_string == eqn._repr_latex_()


@pytest.mark.parametrize(
    "val, expected_str, expected_latex",
    [
        (Variable(1, scale=1000, units="mm"), "1000mm", "$1000\\text{mm}$"),
        (
            Variable(1, scale=1000, units="mm", fmt_string=".2f"),
            "1000.00mm",
            "$1000.00\\text{mm}$",
        ),
        (
            Variable(1, scale=1000, units="mm", fmt_string=".2e"),
            "1.00e+03mm",
            "$1.00 \\times 10^{3}\\text{mm}$",
        ),
        (Variable(1, scale=1000, symbol="a"), "a=1000", "$\\text{a} = 1000$"),
        (
            Variable(1, scale=1000, symbol="a", units="mm"),
            "a=1000mm",
            "$\\text{a} = 1000\\text{mm}$",
        ),
        (
            Variable(1, scale=1000, symbol="a", units="mm", fmt_string=".2f"),
            "a=1000.00mm",
            "$\\text{a} = 1000.00\\text{mm}$",
        ),
    ],
)
def test_scale(val, expected_str, expected_latex):
    assert str(val) == expected_str
    assert val.latex_string == expected_latex


@pytest.mark.parametrize(
    "val, fmt_string, expected",
    [
        (1, ".3e", f"{1:.3e}"),  # test it backs out and uses the default format
        (1, ".3f", f"{1:.3f}"),  # test it backs out and uses the default format
        (1, ".3g", f"{1:.3g}"),  # test it backs out and uses the default format
        (1, ".3%", f"{1:.3%}"),  # test it backs out and uses the default format
        (0.0000001, ".3j", "100e-09"),
        (0.000001, ".3j", "1.00e-06"),
        (0.00001, ".3j", "10.0e-06"),
        (0.0001, ".3j", "100e-06"),
        (0.001, ".3j", "0.00100"),
        (0.01, ".3j", "0.0100"),
        (0.1, ".3j", "0.100"),
        (1, ".3j", "1.00"),
        (10, ".3j", "10.0"),
        (100, ".3j", "100"),
        (1000, ".3j", "1000"),
        (10000, ".3j", "10.0e+03"),
        (100000, ".3j", "100e+03"),
        (1000000, ".3j", "1.00e+06"),
        (-0.0000001, ".3j", "-100e-09"),
        (-0.000001, ".3j", "-1.00e-06"),
        (-0.00001, ".3j", "-10.0e-06"),
        (-0.0001, ".3j", "-100e-06"),
        (-0.001, ".3j", "-0.00100"),
        (-0.01, ".3j", "-0.0100"),
        (-0.1, ".3j", "-0.100"),
        (-1, ".3j", "-1.00"),
        (-10, ".3j", "-10.0"),
        (-100, ".3j", "-100"),
        (-1000, ".3j", "-1000"),
        (-10000, ".3j", "-10.0e+03"),
        (-100000, ".3j", "-100e+03"),
        (-1000000, ".3j", "-1.00e+06"),
        (0.0000001, ".2j", "100e-09"),
        (0.000001, ".2j", "1.0e-06"),
        (0.00001, ".2j", "10e-06"),
        (0.0001, ".2j", "100e-06"),
        (0.001, ".2j", "0.0010"),
        (0.01, ".2j", "0.010"),
        (0.1, ".2j", "0.10"),
        (1, ".2j", "1.0"),
        (10, ".2j", "10"),
        (100, ".2j", "100"),
        (1000, ".2j", "1000"),
        (10000, ".2j", "10e+03"),
        (100000, ".2j", "100e+03"),
        (1000000, ".2j", "1.0e+06"),
        (-0.0000001, ".2j", "-100e-09"),
        (-0.000001, ".2j", "-1.0e-06"),
        (-0.00001, ".2j", "-10e-06"),
        (-0.0001, ".2j", "-100e-06"),
        (-0.001, ".2j", "-0.0010"),
        (-0.01, ".2j", "-0.010"),
        (-0.1, ".2j", "-0.10"),
        (-1, ".2j", "-1.0"),
        (-10, ".2j", "-10"),
        (-100, ".2j", "-100"),
        (-1000, ".2j", "-1000"),
        (-10000, ".2j", "-10e+03"),
        (-100000, ".2j", "-100e+03"),
        (-1000000, ".2j", "-1.0e+06"),
        (0.0000001, ".1j", "100e-09"),
        (0.000001, ".1j", "1e-06"),
        (0.00001, ".1j", "10e-06"),
        (0.0001, ".1j", "100e-06"),
        (0.001, ".1j", "0.001"),
        (0.01, ".1j", "0.01"),
        (0.1, ".1j", "0.1"),
        (1, ".1j", "1"),
        (10, ".1j", "10"),
        (100, ".1j", "100"),
        (1000, ".1j", "1000"),
        (10000, ".1j", "10e+03"),
        (100000, ".1j", "100e+03"),
        (1000000, ".1j", "1e+06"),
        (-0.0000001, ".1j", "-100e-09"),
        (-0.000001, ".1j", "-1e-06"),
        (-0.00001, ".1j", "-10e-06"),
        (-0.0001, ".1j", "-100e-06"),
        (-0.001, ".1j", "-0.001"),
        (-0.01, ".1j", "-0.01"),
        (-0.1, ".1j", "-0.1"),
        (-1, ".1j", "-1"),
        (-10, ".1j", "-10"),
        (-100, ".1j", "-100"),
        (-1000, ".1j", "-1000"),
        (-10000, ".1j", "-10e+03"),
        (-100000, ".1j", "-100e+03"),
        (-1000000, ".1j", "-1e+06"),
        (0.000000123456, ".3j", "123e-09"),
        (0.00000123456, ".3j", "1.23e-06"),
        (0.0000123456, ".3j", "12.3e-06"),
        (0.000123456, ".3j", "0.000123"),
        (0.00123456, ".3j", "0.00123"),
        (0.0123456, ".3j", "0.0123"),
        (0.123456, ".3j", "0.123"),
        (1.23456, ".3j", "1.23"),
        (12.3456, ".3j", "12.3"),
        (123.456, ".3j", "123"),
        (1234.56, ".3j", "1230"),
        (12345.6, ".3j", "12.3e+03"),
        (123456.7, ".3j", "123e+03"),
        (1234567.8, ".3j", "1.23e+06"),
        (12345678.9, ".3j", "12.3e+06"),
        (123456789.0, ".3j", "123e+06"),
        (1234567890.1, ".3j", "1.23e+09"),
        (-0.000000123456, ".3j", "-123e-09"),
        (-0.00000123456, ".3j", "-1.23e-06"),
        (-0.0000123456, ".3j", "-12.3e-06"),
        (-0.000123456, ".3j", "-0.000123"),
        (-0.00123456, ".3j", "-0.00123"),
        (-0.0123456, ".3j", "-0.0123"),
        (-0.123456, ".3j", "-0.123"),
        (-1.23456, ".3j", "-1.23"),
        (-12.3456, ".3j", "-12.3"),
        (-123.456, ".3j", "-123"),
        (-1234.56, ".3j", "-1230"),
        (-12345.6, ".3j", "-12.3e+03"),
        (-123456.7, ".3j", "-123e+03"),
        (-1234567.8, ".3j", "-1.23e+06"),
        (-12345678.9, ".3j", "-12.3e+06"),
        (-123456789.0, ".3j", "-123e+06"),
        (-1234567890.1, ".3j", "-1.23e+09"),
        (1e-6, ".3j", "1.00e-06"),
        (1e-9, ".3j", "1.00e-09"),
        (-1e-6, ".3j", "-1.00e-06"),
        (-1e-9, ".3j", "-1.00e-09"),
        (1e6, ".3J", "1.00E+06"),  # confirm that a capital J gives a capital E
    ],
)
def test_format_sig_fig(val, fmt_string, expected):
    assert format_sig_figs(val, fmt_string) == expected


@given(
    st.floats(min_value=1, max_value=9),
    st.integers(min_value=4, max_value=150).filter(lambda x: x % 3 == 0),
    st.integers(min_value=1, max_value=10),
)
def test_format_sig_fig_hypothesis(mantissa, exponent, sig_figs):
    """
    Test to try and confirm if the output of format_sig_figs matches the formatting
    of the 'e' option in python's default formatter.
    """

    val = mantissa * 10**exponent
    fmt = f".{sig_figs}j"
    assert format_sig_figs(val, fmt) == f"{val:.{sig_figs - 1}e}"


@pytest.mark.parametrize("x", [(1), ("a"), (["a", "b", "c"]), ("$3"), ("3$")])
def test_incl_dollars(x):
    """
    Test the incl_dollars option adds or removes '$$' from the string.
    """

    val_true = Variable(x, incl_dollars=True)
    val_false = Variable(x, incl_dollars=False)

    assert val_true.latex_string[0] == "$"
    assert val_true.latex_string[-1] == "$"
    assert val_false.latex_string[0] != "$"
    assert val_false.latex_string[-1] != "$"

    assert val_true.latex_string != val_false.latex_string
    assert val_true.latex_string[1:-1] == val_false.latex_string
