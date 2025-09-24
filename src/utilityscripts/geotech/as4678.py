"""
To contain functions for working with AS4678.
"""

from math import atan, radians, tan


def s5_2_2_phi_star(*, phi_u_phi: float, phi: float, use_radians: bool = False):
    """
    Calculate the design internal shear friction factor.

    Parameters
    ----------
    phi_u_phi : float
        The uncertainty factor for phi.
    phi : float
        The internal friction angle.

    returns
    -------
        phi* in radians.
    """

    if not use_radians:
        phi = radians(phi)

    return atan(phi_u_phi * tan(phi))
