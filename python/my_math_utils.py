import math


def round_nearest(x, a):
    """
    Round a number to the nearest multiple.

    :param x: the number to round.
    :param a: the multiple to round to.
    """

    return round(round(x / a) * a, -int(math.floor(math.log10(a))))


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

    # figure out the power we need to get the correct no. of significant figures
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

        print(f"x: {x}, a: {a}, round_nearest={round_nearest(x,a)}")

    print()
    print("Done")
    print("=" * 10)
    print()
