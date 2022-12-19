"""
Contains some methods for working with geotech properties.
"""

from math import exp, pi, radians, tan


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
