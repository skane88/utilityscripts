"""
Some basic math utilities
"""

import math
from decimal import Decimal


def num_digits(number, max_digits=100):
    """
    Determine the number of digits of precision in a number.

    :param number: the number to get the precision of.
    :max_digits: the maximum number of digits.
        NOTE: necessary because floating point math may
        result in the tests in this function never returning
        an equality.
    """

    for i in range(max_digits):
        digits = -int(math.floor(math.log10(number))) + i
        if round(number, digits) == number:
            break
    return digits


def m_round(x, base):
    """
    Round a number to the nearest multiple.

    Based on code at:
    https://stackoverflow.com/questions/28425705/python-round-a-float-to-nearest-0-05-or-to-multiple-of-another-float

    :param x: the number to round.
    :param base: the multiple to round to.
    """

    base = abs(base)

    # next need to find the number of digits in the base.
    # this is because sometimes round(x / base) * base
    # returns a floating point number slightly above or below
    # the real result. It is therefore necessary to round
    # the resulting value to the number of digits of
    # base.
    frac_digits = num_digits(base)

    return round(round(x / base) * base, frac_digits)


def m_floor(x, base):
    """
    Floor a number to the nearest multiple.

    :param x: the number to round.
    :param base: the multiple to round to.
    """

    x = Decimal(str(x))
    base = Decimal(str(base))
    base = abs(base)

    # next need to find the number of digits in the base.
    # this is because sometimes round(x / base) * base
    # returns a floating point number slightly above or below
    # the real result. It is therefore necessary to round
    # the resulting value to the number of digits of
    # base.
    frac_digits = num_digits(base)

    return float(round(math.floor(x / base) * base, frac_digits))


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


if __name__ == "__main__":
    round_pairs = [(11.1212341234, 0.25), (-12.34, 0.5), (0.0623413, 0.1)]

    print()
    print("Testing round_nearest")
    print()
    for x, a in round_pairs:
        print(f"x: {x}, a: {a}, round_nearest={m_round(x, a)}")

    print()
    print("Done")
    print("=" * 10)
    print()
