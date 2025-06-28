"""
Some basic math utilities
"""

import math
from decimal import Decimal
from math import log10


def m_round(x, base):
    """
    Round a number to the nearest multiple.

    Based on code at:
    https://stackoverflow.com/questions/28425705/python-round-a-float-to-nearest-0-05-or-to-multiple-of-another-float

    :param x: the number to round.
    :param base: the multiple to round to.
    """

    x, base = Decimal(str(x)), Decimal(str(base))

    return float(round(x / base) * base)


def m_floor(x, base, *, float_tolerance: float | None = 1e-6):
    """
    Floor a number to the nearest multiple.

    Note: due to floating point math, it is necessary to
    convert the numbers to Decimals first. This
    will incur a performance overhead so only use this
    method if there is no better way to do this function.

    Parameters
    ----------
    x : float
        The number to round.
    base : float
        The multiple to round to.
    float_tolerance : float, optional
        A tolerance for floating point math. If not None, the function first
        rounds to the float_tolerance and then completes the rounding.
    """

    if float_tolerance is not None:
        x = m_round(x, float_tolerance)

    x, base = Decimal(str(x)), Decimal(str(base))
    x_orig = x

    val = math.floor(x / base) * base

    if val > x_orig:
        val = val - abs(base)

    return float(val)


def m_ceil(x, base, *, float_tolerance: float | None = 1e-6):
    """
    Floor a number to the nearest multiple.

    Note: due to floating point math, it is necessary to
    convert the numbers to Decimals first. This
    will incur a performance overhead so only use this
    method if there is no better way to do this function.

    Parameters
    ----------
    x : float
        The number to round.
    base : float
        The multiple to round to.
    float_tolerance : float, optional
        A tolerance for floating point math. If not None, the function first
        rounds to the float_tolerance and then completes the rounding.
    """

    if float_tolerance is not None:
        x = m_round(x, float_tolerance)

    x, base = Decimal(str(x)), Decimal(str(base))
    x_orig = x
    val = math.ceil(x / base) * base

    if val < x_orig:
        val = val + abs(base)

    return float(val)


def round_significant(x, s: int = 3):
    """
    Round a number to a specified significant figure.

    Based on an answer at

    https://stackoverflow.com/a/9557559

    :param x: The number to round.
    :param s: The significance to round to.
    """

    if x == 0:
        # bail out early on 0.
        return 0

    # can't take a log of a -ve number, so have to abs x
    xsign = x / abs(x)
    x = abs(x)

    # get the exponent in powers of 10s
    max10exp = math.floor(math.log10(x))

    # figure out the power we need to get
    # the correct no. of significant figures
    sig10pow = 10 ** (max10exp - s + 1)

    # raie the number to have s digits past the decimal point
    floated = x * 1.0 / sig10pow

    # drop it back down to the original power of 10 & return
    return round(floated) * sig10pow * xsign


def sci_not(value: float) -> tuple[float, int]:
    """
    Convert a number to a tuple of (mantissa, exponent) in scientific notation.
    """

    if value == 0:
        return 0, 0

    sign = -1 if value < 0 else 1
    value = abs(value)

    exponent = int(log10(value))
    mantissa = value / 10**exponent

    return sign * mantissa, exponent
