"""
This file contains some interpolation functions.
"""

import timeit
from typing import List, Union

import numpy as np


def multi_interp(
    *,
    x: Union[float, List[float], np.ndarray],
    xp: Union[List[float], np.ndarray],
    fp: Union[List[float], List[List[float]], np.ndarray],
    left: float | None = None,
    right: float | None = None,
) -> Union[float, np.ndarray]:
    """
    Implements an equivalent of np.interp that will interpolate through multiple
    sets of y points.

    NOTE: this is NOT 2D interpolation. It is simply repeated 1d interpolation of the
    same x interpolation through multiple rows of y values. If fp has a single row, this
    is equivalent to calling the numpy function np.interp.

    :param x: The x points to interpolate.
    :param xp: The original x data points. Must be sorted. Raises a value error if not
        sorted.
    :param fp: The original y data points to interpolate.
    :param left: An optional fill value to use if a value in x is less than the values
        in xp (and thus cannot be interpolated). If None, the value of fp at min(xp)
        will be used.
    :param right: An optional fill value to use if a value in x is greater than the
        values in xp (and thus cannot be interpolated). If None, the value of fp at
        max(xp) will be used.
    """

    if x is None:
        raise ValueError(f"Expected a value to interpolate. Actual x provided: x={x}")
    if x == []:
        raise ValueError(f"Expected a value to interpolate. Actual x provided: x={x}")

    x = np.array(x)
    fp = np.array(fp)

    if len(fp.shape) == 1:
        return np.interp(x=x, xp=xp, fp=fp, left=left, right=right)

    return np.vstack([np.interp(x=x, xp=xp, fp=f, left=left, right=right) for f in fp])


if __name__ == "__main__":
    x = [-0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50]

    xp = [0, 1, 2]

    fp = [[1, 2, 3], [11, 12, 13]]

    print(multi_interp(x=x, xp=xp, fp=fp))

    def wrapper(func, *args, **kwargs):
        """
        Helper function to allow us to test the multi_interp function
        with timeit.
        """

        def inner_func():
            return func(*args, **kwargs)

        return inner_func

    wrapped = wrapper(multi_interp, x=x, xp=xp, fp=fp)

    NUMBER = 10000
    ti = timeit.Timer(wrapped)

    time_info = ti.repeat(repeat=10, number=NUMBER)

    print()
    print(f"Average time to run multi_interp: {1e6 * min(time_info) / NUMBER} us")
    print()
    print("Done")
    print("=" * 88)
