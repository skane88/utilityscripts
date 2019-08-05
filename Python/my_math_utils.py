import math


def round_nearest(x, a):
    """
    Round a number to the nearest multiple.

    :param x: the number to round.
    :param a: the multiple to round to.
    """

    return round(round(x / a) * a, -int(math.floor(math.log10(a))))


if __name__ == "__main__":

    round_pairs = [(11.1212341234, 0.25), (-12.34, 0.5), (0.0623413, 0.1)]

    print()
    print("Testing round_nearest")
    print()
    for x, a in round_pairs:

        print(f"x: {x}, a: {a}, round_nearest={round_nearest(x,a)}")

    print()
    print("Done")
    print("="*10)
    print()
