"""
Very basic integration of numpy arrays.
There are probably better ways to do this in numpy...
"""

import numpy as np


def integrate(x: np.ndarray, y: np.ndarray):
    """
    Integrate 2x numpy arrays.

    :param x: The x points of the function to integrate.
    :param y: The y points of the function to integrate.
    """

    x0 = x[:-1]
    x1 = x[1:]
    y0 = y[:-1]
    y1 = y[1:]

    delta = x1 - x0

    return np.sum(((y0 + y1) / 2) * delta)
