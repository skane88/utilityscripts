"""
Python file to determine steel design properties based on AS4600
"""

from math import pi


def f_cr_plate(k, elastic_modulus, thickness, width, poisson_ratio=0.3):
    """
    Calculate the critical buckling stress of a plate element.

    Parameters
    ----------
    k : float
        The plate buckling coefficient.
    elastic_modulus : float
        The elastic modulus.
    thickness : float
        The thickness of the element.
    width : float
        The width of the element.
    poisson_ratio : float, optional
        Poisson's ratio. Default is 0.3.

    Returns
    -------
    float
        The critical buckling stress of the plate element.
    """

    return ((k * pi**2 * elastic_modulus) / (12 * (1 - poisson_ratio**2))) * (
        thickness / width
    ) ** 2


def slenderness_ratio(f_star, f_cr):
    """
    Calculate the slenderness ratio of an element as per AS4600 S2.2.1.2

    Parameters
    ----------
    f_star : float
        The design compressive stress.
    f_cr : float
        The critical buckling stress.

    Returns
    -------
    float
        The slenderness ratio.
    """
    return (f_star / f_cr) ** 0.5


def rho(slenderness):
    """
    Calculate the effective width factor for the given slenderness ratio.

    Parameters
    ----------
    slenderness : float
        The slenderness ratio of the element.

    Returns
    -------
    float
        The effective width factor.
    """
    return min(1.0, (1 - 0.22 / slenderness) / slenderness)
