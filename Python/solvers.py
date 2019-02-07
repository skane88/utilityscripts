"""
Implements the bisection method.
"""

from sys import float_info


def test_func(x):

    return x * 2 / 3 - 1


def sign(num):
    """
    Check the sign of a number. Returns -1 if less than 0, otherwise returns 1.
    """

    if num < 0:
        return -1
    else:
        return 1


def bisection(
    func,
    x_low: float = float_info.min,
    x_high: float = float_info.max,
    tol: float = 1e-10,
):
    """
    Implements the bi-section method.

    Guaranteed to find a root if one exists between the guesses. If more than one root
    exists though there is no guarantee about which one will be returned.

    :param func: A function with a single input parameter ('x') to be solved for 0.
    :param x_low: The lower bounds of the range to check.
    :param x_high: The upper bounds of the range to check.
    :param tol: The solution tolerance.
    :returns: Returns a tuple: (root, no. of iterations)
    """

    if x_high == x_low:
        raise ValueError("Expected guesses to be different.")

    i = 0

    while abs(x_high - x_low) > tol and x_high != x_low:

        y_low = func(x_low)
        y_high = func(x_high)

        if sign(y_low) == sign(y_high):
            raise ValueError("Expected the guesses to bracket the root")

        x_mid = (x_low + x_high) / 2
        y_mid = func(x_mid)

        if sign(y_low) == sign(y_mid):
            x_low = x_mid
        else:
            x_high = x_mid

        i += 1

    return (x_low + x_high) / 2, i


if __name__ == "__main__":

    print("Test")
    print(
        f"Solution is: {bisection(test_func)[0]} in {bisection(test_func)[1]} iterations"
    )
    input("Press any key to exit")
