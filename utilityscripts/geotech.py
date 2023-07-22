"""
Contains some methods for working with geotech properties.
"""

import warnings
from math import exp, pi, radians, tan

import numpy as np


def q_ult(
    c,
    N_c,
    gamma,
    q,
    N_q,
    B,
    N_gamma,
    s_c=1.0,
    s_q=1.0,
    s_gamma=1.0,
    i_c=1.0,
    i_q=1.0,
    i_gamma=1.0,
):
    """
    Calculate the bearing capacity of a footing.

    :param c: The cohesion of the soil.
    :param N_c: The bearing capacity factor for soil shear strength.
    :param s_c: The shape factor for the soil shear strength.
    :param i_c: The load inclination factor for the soil shear strength.
    :param gamma: The density of the soil.
    :param q: The net vertical pressure of any soil above the level of the footing,
        considering water level etc., and also any additional surcharge load that will
        act to prevent displacement of the soil wedge.
        NOTE 1: The surcharge for soil above the footing should consider the risk of
            the surface level changing due to local excavations etc. around the footing.
        NOTE 2: additional surcharge should only be included if it can be guaranteed to
            be present in the load cases for which q_ult is being calculated.
            Additionally, consideration should be given to factoring the load down as
            appropriate for a stabilising load component.
    :param N_q: The bearing capacity factor for the surcharge load on the footing.
    :param s_q: The shape factor for the surcharge load on the footing.
    :param i_q: The load inclination factor for the surcharge load on the footing.
    :param B: The width of the footing. Note that if the load is eccentric
        the width should be reduced to the effective width.
    :param N_gamma: The bearing capacity factor for the soil below the footing.
    :param s_gamma: The shape factor for the soil below the footing.
    :param i_gamma: The load inclination factor for the soil below the footing.
    """

    return (
        c * N_c * s_c * i_c
        + q * N_q * s_q * i_q
        + 0.5 * gamma * B * N_gamma * s_gamma * i_gamma
    )


def N_q(phi):
    """
    Return the bearing capacity factor for the surcharge of material above the
    foundation as per Meyerhoff

    :param phi: The shear friction angle of the material.
    """

    return exp(pi * tan(phi)) * (tan(radians(45) + phi / 2)) ** 2


def N_c(N_q, phi):
    """
    Return the bearing capacity factor for the shear strength of the soil, as per
    Meyerhoff.

    :param N_q: The bearing capacity factor for surcharge.
    :param phi: THe shear friction angle of the material.
    """

    return (N_q - 1) / tan(phi)


def N_gamma_Meyerhoff(N_q, phi):
    """
    Return the bearing capacity factor for soil below the footing as per Meyerhoff.

    :param N_q: The bearing capacity factor for the shear strength of the soil
    :param phi: The shear friction angle of the material.
    """

    return (N_q - 1) * tan(1.4 * phi)


def N_gamma_Hansen(N_q, phi):
    """
    Return the bearing capacity factor for soil below the footing as per Hansen.

    :param N_q: The bearing capacity factor for the shear strength of the soil
    :param phi: The shear friction angle of the material.
    """

    return 1.8 * (N_q - 1) * tan(phi)


def N_gamma_EC7(N_q, phi):
    """
    Return the bearing capacity factor for soil below the footing as per EC7.

    :param N_q: The bearing capacity factor for the shear strength of the soil
    :param phi: The shear friction angle of the material.
    """

    return 2.0 * (N_q - 1) * tan(phi)


def boussinesq_point_sigma_z(Q, r, z):
    """
    Determine the vertical stresses due to a point load on the surface according to
    Boussinesq's method.

    :param Q: The point load.
    :param r: The radial distance between the point load and the point at which
        stresses are to be determined.
    :param z: The depth between the surface and the point at which stresses are
        required.
    """

    a = (3 * Q) / (2 * pi * z**2)
    b = (1 / (1 + (r / z) ** 2)) ** (5 / 2)

    return a * b


def boussinesq_point_sigma_r(Q, r, z, v):
    """
    Determine the horizontal stresses due to a point load on the surface according to
    Boussinesq's method. These stresses are in the radial direction away from the
    point load.

    Note that the method assumes an infinite field for these stresses to develop in.
    For applications to retaining walls the wall itself cuts off this infinite field.
    Some references suggest that the pressures on the wall may need to be multiplied
    by 2x to account for this effect.

    :param Q: The point load.
    :param r: The radial distance between the point load and the point at which
        stresses are to be determined.
    :param z: The depth between the surface and the point at which stresses are
        required.
    :param v: Poisson's ratio for the soil. At v=0.5, there is no tangential stress.
    """

    a = Q / (2 * pi)

    b = (3 * z * r**2) / ((r**2 + z**2) ** (5 / 2))
    c = (1 - 2 * v) / (r**2 + z**2 + z * ((r**2 + z**2) ** (1 / 2)))

    d = b - c

    return a * d


def boussinesq_point_sigma_theta(Q, r, z, v):
    """
    Determine the horizontal stresses due to a point load on the surface according to
    Boussinesq's method. These stresses are in the tangential direction around the
    point load.

    Note that the method assumes an infinite field for these stresses to develop in.
    For applications to retaining walls the wall itself cuts off this infinite field.
    Some references suggest that the pressures on the wall may need to be multiplied
    by 2x to account for this effect.

    :param Q: The point load.
    :param r: The radial distance between the point load and the point at which
        stresses are to be determined.
    :param z: The depth between the surface and the point at which stresses are
        required.
    :param v: Poisson's ratio for the soil. At v=0.5, there is no tangential stress.
    """

    a = Q / (2 * pi)
    b = 1 - 2 * v
    c = z / ((r**2 + z**2) ** (3 / 2))
    d = 1 / (r**2 + z**2 + z * ((r**2 + z**2) ** (1 / 2)))

    e = c - d

    return a * b * e


def boussinesq_point_tau_rz(Q, r, z):
    """
    Determine the vertical shear stresses due to a point load on the surface according
    to Boussinesq's method. These shear stresses are directed radially outwards from
    the point load towards the point of assessment.

    Note that the method assumes an infinite field for these stresses to develop in.
    For applications to retaining walls the wall itself cuts off this infinite field.
    Some references suggest that the pressures on the wall may need to be multiplied
    by 2x to account for this effect.

    :param Q: The point load.
    :param r: The radial distance between the point load and the point at which
        stresses are to be determined.
    :param z: The depth between the surface and the point at which stresses are
        required.
    """

    a = (3 * Q) / (2 * pi)
    b = (r * z**2) / ((r**2 + z**2) ** (5 / 2))

    return a * b


def boussinesq_point_sigma_x(Q, x, y, z, v):
    """
    Determine the horizontal stresses due to a point load on the surface according to
    Boussinesq's method. These stresses are in the "x" direction where x and y are
    defined as follows:

    * x is perpendicular to a vertical plane through the soil (e.g. a retaining wall)
    * y is parallel to the vertical plane.

    Note 1: the method assumes an infinite field for these stresses to develop in.
    For applications to retaining walls the wall itself cuts off this infinite field.
    Some references suggest that the pressures on the wall may need to be multiplied
    by 2x to account for this effect.

    Note 2: this method ignores sigma_theta as per J Bowles 5ed fig 11-16.

    :param Q: The point load.
    :param x: the distance from the point load perpendicular to a vertical plane through
       the soil (e.g. a retaining wall)
    :param y: the distance between the point load and the point of interest on the
        vertical plane.
    :param z: The depth between the surface and the point at which stresses are
        required.
    :param v: Poisson's ratio for the soil. At v=0.5, there is no tangential stress.
    """

    r = (x**2 + y**2) ** (1 / 2)

    sigma_r = boussinesq_point_sigma_r(Q=Q, r=r, z=z, v=v)

    x_ratio = 1 if x == 0 and r == 0 else x / r
    return sigma_r * x_ratio


def boussinesq_patch_sigma_z(q, x, y, z, x_patch, y_patch, n_int_x=10, n_int_y=10):
    """
    Determine the vertical stresses due to a rectangular patch load on the surface
    according to Boussinesq's method.
    Calculated by integrating a number of finite points.

    * x is perpendicular to a vertical plane through the soil (e.g. a retaining wall)
    * y is parallel to the vertical plane.

    Note 1: the method assumes an infinite field for these stresses to develop in.
    For applications to retaining walls the wall itself cuts off this infinite field.
    Some references suggest that the pressures on the wall may need to be multiplied
    by 2x to account for this effect.

    :param q: The applied pressure.
    :param x: the distance from the centroid of the load patch to a vertical plane
        through the soil (e.g. a retaining wall)
    :param y: the distance from the centroid of the load patch and the point of interest
        on the vertical plane.
    :param z: The depth between the surface and the point at which stresses are
        required.
    :param x_patch: the width of the load patch in the x direction.
    :param y_patch: the width of the load patch in the y direction.
    :param n_int_x: The number of elements to break the load into in the z direction.
    :param n_int_y: The number of elements to break the load into in the y direction.
    """

    if (x_patch / 2) > x:
        warnings.warn("Load patch crosses over the plane of application")

    Q_total = q * (x_patch * y_patch)
    Q = Q_total / (n_int_y * n_int_x)

    x_points = np.linspace(x - x_patch / 2, x + x_patch / 2, n_int_x + 1)
    x_points = (x_points[1:] + x_points[:-1]) / 2

    y_points = np.linspace(y - y_patch / 2, y + y_patch / 2, n_int_y + 1)
    y_points = (y_points[1:] + y_points[:-1]) / 2

    sigma_z = 0

    for xi in x_points:
        for yi in y_points:
            r = (xi**2 + yi**2) ** (1 / 2)

            sigma_z += boussinesq_point_sigma_z(Q=Q, r=r, z=z)

    return sigma_z


def boussinesq_patch_sigma_x(q, x, y, z, v, x_patch, y_patch, n_int_x=10, n_int_y=10):
    """
    Determine the horizontal stresses due to a rectangular patch load on the surface
    according to Boussinesq's method. Stresses are calculated on a vertical plane
    through the soil.
    Calculated by integrating a number of finite points.

    * x is perpendicular to a vertical plane through the soil (e.g. a retaining wall)
    * y is parallel to the vertical plane.

    Note 1: the method assumes an infinite field for these stresses to develop in.
    For applications to retaining walls the wall itself cuts off this infinite field.
    Some references suggest that the pressures on the wall may need to be multiplied
    by 2x to account for this effect.

    Note 2: this method ignores sigma_theta as per J Bowles 5ed fig 11-16.

    Note 3: If the patch overlaps the point x, then some loads will cancel each other
    out. This may result in incorrect pressure distributions (you may need to consider
    splitting the footing in half to determine the pressure accurately).

    :param q: The applied pressure.
    :param x: the distance from the centroid of the load patch to a vertical plane
        through the soil (e.g. a retaining wall)
    :param y: the distance from the centroid of the load patch and the point of interest
        on the vertical plane.
    :param z: The depth between the surface and the point at which stresses are
        required.
    :param v: Poisson's ratio for the soil. At v=0.5, there is no tangential stress.
    :param x_patch: the width of the load patch in the x direction.
    :param y_patch: the width of the load patch in the y direction.
    :param n_int_x: The number of elements to break the load into in the z direction.
    :param n_int_y: The number of elements to break the load into in the y direction.
    """

    if (x_patch / 2) > x:
        warnings.warn("Load patch crosses over the plane of application")

    Q_total = q * (x_patch * y_patch)
    Q = Q_total / (n_int_y * n_int_x)

    x_points = np.linspace(x - x_patch / 2, x + x_patch / 2, n_int_x + 1)
    x_points = (x_points[1:] + x_points[:-1]) / 2

    y_points = np.linspace(y - y_patch / 2, y + y_patch / 2, n_int_y + 1)
    y_points = (y_points[1:] + y_points[:-1]) / 2

    sigma_x = 0

    for xi in x_points:
        for yi in y_points:
            sigma_x += boussinesq_point_sigma_x(Q=Q, x=xi, y=yi, z=z, v=v)

    return sigma_x
