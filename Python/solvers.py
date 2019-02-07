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
    max_its=None,
):
    """
    Implements the bi-section method.

    Guaranteed to find a root if one exists between the guesses. If more than one root
    exists though there is no guarantee about which one will be returned.

    :param func: A function with a single input parameter ('x') to be solved for 0.
    :param x_low: The lower bounds of the range to check.
    :param x_high: The upper bounds of the range to check.
    :param tol: The solution tolerance.
    :param max_its: A maximum number of iterations to perform. If convergence is not
        achieved within tol when max_its is reached, an error is raised.

        If ``None``, the solver will continue until convergence is reached (potentially
        infinitely, although it is likely that your computer's numerical precision will
        result in convergence before an infinite number of iterations is reached)
    :returns: Returns a tuple: (root, no. of iterations)
    """

    if x_high == x_low:
        raise ValueError("Expected guesses to be different.")

    if max_its is not None:
        if max_its <= 1:
            raise ValueError("Maximum no. of iterations should be > 1")

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

        if max_its is not None:
            if i > max_its:
                raise ValueError(
                    f"Exceeded maximum number of iterations. "
                    + f"Current root approximation is {x_mid}."
                )

    return (x_low + x_high) / 2, i


if __name__ == "__main__":

    print("Test")
    print(
        f"Solution is: {bisection(test_func)[0]} in {bisection(test_func)[1]} iterations"
    )
    input("Press any key to exit")
