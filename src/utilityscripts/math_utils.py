"""
Some basic math utilities
"""

import math
import sys
from decimal import Decimal
from fractions import Fraction
from numbers import Real


def m_round(x: Real, base: Real) -> float:
    """
    Round a number to the nearest multiple.

    Notes
    -----
    - Based on code at:
        https://stackoverflow.com/questions/28425705/python-round-a-float-to-nearest-0-05-or-to-multiple-of-another-float

    Parameters
    ----------
    x : Real
        the number to round.
    base : Real
        the multiple to round to.
    """

    if isinstance(x, Decimal) and (x.is_nan() or x.is_infinite()):
        raise ValueError(f"Cannot round {x}.")

    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"Cannot round {x}.")

    if base == 0:
        raise ValueError(f"Cannot round to a multiple of {base}.")

    if x == 0:
        return 0

    x = (
        Decimal(x.numerator) / Decimal(x.denominator)
        if isinstance(x, Fraction)
        else Decimal(x)
    )
    base = Decimal(str(base))

    return float(round(x / base) * base)


def m_floor(x: Real, base: Real, *, float_tolerance: float | None = 1e-6) -> float:
    """
    Floor a number to the nearest multiple.

    Notes
    -----
    - Due to floating point math, it is necessary to convert the numbers to Decimals
        first. This will incur a performance overhead so only use this method if
        there is no better way to do this function.

    Parameters
    ----------
    x : Real
        The number to round.
    base : Real
        The multiple to round to.
    float_tolerance : float, optional
        A tolerance for floating point math. If not None, the function first
        rounds to the float_tolerance and then completes the rounding.
    """

    if isinstance(x, Decimal) and (x.is_nan() or x.is_infinite()):
        raise ValueError(f"Cannot round {x}.")

    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"Cannot round {x}.")

    if base == 0:
        raise ValueError(f"Cannot round to a multiple of {base}.")

    if float_tolerance is not None:
        x = m_round(x, float_tolerance)

    x, base = Decimal(str(x)), Decimal(str(base))
    x_orig = x

    val = math.floor(x / base) * base

    if val > x_orig:
        val = val - abs(base)

    return float(val)


def m_ceil(x: Real, base: Real, *, float_tolerance: float | None = 1e-6) -> float:
    """
    Floor a number to the nearest multiple.

    Notes
    -----
    - Due to floating point math, it is necessary to convert the numbers to Decimals
        first. This will incur a performance overhead so only use this method if there
        is no better way to do this function.

    Parameters
    ----------
    x : Real
        The number to round.
    base : Real
        The multiple to round to.
    float_tolerance : float, optional
        A tolerance for floating point math. If not None, the function first
        rounds to the float_tolerance and then completes the rounding.
    """

    if isinstance(x, Decimal) and (x.is_nan() or x.is_infinite()):
        raise ValueError(f"Cannot round {x}.")

    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"Cannot round {x}.")

    if base == 0:
        raise ValueError(f"Cannot round to a multiple of {base}.")

    if float_tolerance is not None:
        x = m_round(x, float_tolerance)

    x, base = Decimal(str(x)), Decimal(str(base))
    x_orig = x
    val = math.ceil(x / base) * base

    if val < x_orig:
        val = val + abs(base)

    return float(val)


def round_significant(x: Real, s: int = 3):
    """
    Round a number to a specified significant figure.

    Based on an answer at

    https://stackoverflow.com/a/9557559

    Notes
    -----
    - The function uses python's Decimal class to do mathematics, rounding etc. to try
        and avoid floating point errors.

    Parameters
    ----------
    x : Real
        The number to round.
    s : int
        The significance to round to.
    """

    type_in = type(x)

    if x == 0:
        # bail out early on 0.
        return 0

    x = Decimal(x)

    # can't take a log of a -ve number, so have to abs x
    xsign = 1 if x > 0 else -1
    x = abs(x)

    # get the exponent in powers of 10s
    max10exp = Decimal(math.floor(x.log10()))

    # figure out the power we need to get
    # the correct no. of significant figures
    sig10pow = 10 ** (max10exp - s + 1)

    # raise the number to have s digits past the decimal point
    floated = x * Decimal("1.0") / sig10pow

    # drop it back down to the original power of 10 & return
    rounded = round(floated) * sig10pow * xsign

    # convert back to the original type
    return type_in(rounded)


def scientific_number(value: Real) -> tuple[float, int]:
    """
    Convert a number to a tuple of (mantissa, exponent) in scientific notation.

    Parameters
    ----------
    value : Real
        The value to convert.

    Returns
    -------
    tuple[float, int]
        The parts of the number in scientific notation.
    """

    if isinstance(value, Decimal) and (value.is_nan() or value.is_infinite()):
        raise ValueError(f"Cannot convert {value} to scientific notation.")

    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"Cannot convert {value} to scientific notation.")

    if value == 0:
        return 0, 0

    if abs(value) > sys.float_info.max:
        raise ValueError(
            f"{value} is larger than the maximum float value {sys.float_info.max}."
        )

    if abs(value) < sys.float_info.min:
        raise ValueError(
            f"{value} is smaller than the minimum float value {sys.float_info.min}."
        )

    sign = -1 if value < 0 else 1

    value = (
        Decimal(value.numerator) / Decimal(value.denominator)
        if isinstance(value, Fraction)
        else Decimal(value)
    )
    value = abs(value)

    exponent = int(math.log10(value))
    pow_10 = Decimal("10") ** exponent
    mantissa = value / pow_10

    return float(sign * mantissa), exponent


def engineering_number(value: Real) -> tuple[float, int]:
    """
    Convert a number to a tuple of (mantissa, exponent) in engineering notation.
    Engineering notation is similar to scientific notation but limits the exponent
    to powers of 3.

    Parameters
    ----------
    value : Real
        The value to convert.

    Returns
    -------
    tuple[float, int]
        The parts of the number in engineering notation.
    """

    if value == 0:
        return 0, 0

    if abs(value) > sys.float_info.max:
        raise ValueError(
            f"{value} is larger than the maximum float value {sys.float_info.max}."
        )

    if abs(value) < sys.float_info.min:
        raise ValueError(
            f"{value} is smaller than the minimum float value {sys.float_info.min}."
        )

    mantissa, exponent = scientific_number(value)

    if exponent % 3 == 0:
        return mantissa, exponent

    mantissa = Decimal(mantissa)
    offset = exponent % 3

    mantissa = mantissa * Decimal("10") ** offset
    exponent -= offset

    return float(mantissa), exponent
