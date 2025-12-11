"""
To contain methods for analysing floorplate.
"""

from __future__ import annotations

from math import log, pi

import numpy as np


def flat_plate_bending_uniform(*, a: float, b: float, t: float, e: float, q: float):
    """
    Calculate the maximum stress in a flat plate under uniform bending.

    Notes
    -----
    * This is based on Roark's stress & strain [1]_. See Table 11.4 case 1a.
    * This assumes the plate is simply supported on all sides.

    Parameters
    ----------
    a : float
        The length (long side) of the plate. In m.
    b : float
        The width (short side) of the plate. In m.
    t : float
        The thickness of the plate. In m.
    e : float
        The modulus of elasticity of the plate. In Pa.
    q : float
        The uniform load on the plate. In Pa.

    Returns
    -------
    The maximum stress and strain in the plate. In Pa.

    References
    ----------
    .. [1] Roark's Stress & Strain, 8th Ed, Table 11.4
    """

    a_b_ratio = np.asarray(
        [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0, 1000]
    )  # use 1000 for infinity
    beta_data = np.asarray(
        [0.2874, 0.3762, 0.4530, 0.5172, 0.5688, 0.6102, 0.7134, 0.7410, 0.7476, 0.7500]
    )
    alpha_data = np.asarray(
        [0.0444, 0.0616, 0.0770, 0.0906, 0.1017, 0.1110, 0.1335, 0.1400, 0.1417, 0.1421]
    )
    gamma_data = np.asarray(
        [0.420, 0.455, 0.478, 0.491, 0.499, 0.503, 0.505, 0.502, 0.501, 0.500]
    )

    if a < b:  # a_b_ratio needs to be > 1.0
        a, b = b, a

    alpha = np.interp(a / b, a_b_ratio, alpha_data)
    beta = np.interp(a / b, a_b_ratio, beta_data)
    gamma = np.interp(a / b, a_b_ratio, gamma_data)

    sigma_max = beta * q * b**2 / t**2
    y_max = alpha * q * b**4 / (e * t**3)
    r_max = gamma * q * b

    return {"sigma_max": sigma_max, "y_max": y_max, "R_max": r_max}


def flat_plate_bending_point(
    *, a: float, b: float, t: float, e: float, w: float, r_o: float
):
    """
    Calculate the maximum stress in a flat plate under a point load.

    Notes
    -----
    * This is based on Roark's stress & strain [1]_. See table 11.4 case 1b.
    * This assumes the plate is simply supported on all sides.
    * Assumes that poisson's ratio for steel is 0.3.

    Parameters
    ----------
    a : float
        The length (long side) of the plate. In m.
    b : float
        The width (short side) of the plate. In m.
    t : float
        The thickness of the plate. In m.
    e : float
        The modulus of elasticity of the plate. In Pa.
    w : float
        The point load on the plate. In N.
    r_o : float
        The radius of the point load. In m.

    References
    ----------
    .. [1] Roark's Stress & Strain, 8th Ed, Table 11.4
    """

    a_b_ratio = np.asarray([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 1000])
    beta_data = np.asarray([0.435, 0.650, 0.789, 0.875, 0.927, 0.958, 1.000])
    alpha_data = np.asarray([0.1267, 0.1478, 0.1621, 0.1715, 0.1770, 0.1805, 0.1851])

    if a < b:
        a, b = b, a

    alpha = np.interp(a / b, a_b_ratio, alpha_data)
    beta = np.interp(a / b, a_b_ratio, beta_data)

    r_prime_o = (1.6 * r_o**2 + t**2) ** 0.5 - 0.675 * t if r_o < 0.5 * t else r_o

    sigma_max = ((3 * w) / (2 * pi * t**2)) * (
        (1.3) * log((2 * b) / (pi * r_prime_o)) + beta
    )
    y_max = alpha * w * b**2 / (e * t**3)

    return {"sigma_max": sigma_max, "y_max": y_max}
