"""
File to contain functions based on Australian Standard AS4100.
"""

from __future__ import annotations


def alpha_m(*, m_m, m_2, m_3, m_4):
    """
    Determines the moment modification factor as per AS4100 S5.6.1.1.a.iii

    :param m_m: The maximum moment.
    :param m_2: The moment at the 1st 1/4 point.
    :param m_3: The moment at mid-span.
    :param m_4: The moment at the 2nd 1/4 point.
    """

    return 1.7 * m_m / (m_2**2 + m_3**2 + m_4**2) ** 0.5


def k_t(
    *,
    restraint_code: str,
    d_1: float,
    length: float,
    t_f: float,
    t_w: float,
    n_w: float = 1,
):
    """
    Compute the twist restraint factor based on AS4100 Table 5.6.3(a)

    Parameters
    ----------
    restraint_code : str
        Code representing the type of restraint.
    d_1 : float
        The clear web depth.
    length : float
        The length between restraints.
    t_f : float
        The flange thickness.
    t_w : float
        The web thickness.
    n_w : float, optional
        THe number of webs of the beam section.

    Returns
    -------
    float
        Computed twist restraint factor.

    Examples
    --------
    >>> k_t(restraint_code="PP", d_1=0.206, length=4.0, t_f=0.012, t_w=0.0065)
    1.08101

    >>> k_t(restraint_code="FF", d_1=0.206, length=4.0, t_f=0.012, t_w=0.0065)
    1.00000
    """

    if restraint_code in ["FF", "FL", "LF", "LL", "FU", "UF"]:
        return 1.0

    if restraint_code in ["FP", "PF", "PL", "LP", "PU", "UP"]:
        return 1 + ((d_1 / length) * (t_f / (2 * t_w)) ** 3) / n_w

    return 1 + (2.0 * (d_1 / length) * (t_f / (2 * t_w)) ** 3) / n_w


def alpha_v(*, d_p, t_w, f_y, s, f_y_ref=250.0):
    """
    Calculate the stiffened web shear buckling parameter alpha_v as per
    AS4100 S5.11.5.2.

    :param d_p: The depth of the web panel.
    :param t_w: The thickness of the web.
    :param f_y: The yield strength of the web panel.
    :param s: The length of the web or spacing of vertical stiffeners that meet the
        requirement of AS4100.
    :param f_y_ref: The reference yield stress, nominally 250.
    """

    a1 = (82 / ((d_p / t_w) * (f_y / f_y_ref) ** 0.5)) ** 2

    a2_param = 1.0 if (s / d_p) <= 1.0 else 0.75

    a2 = (a2_param / ((s / d_p) ** 2)) + 1.0

    return min(a1 * a2, 1.0)


def alpha_d(*, alpha_v, d_p, s):
    """
    Calculate the stiffened web shear buckling parameter alpha_d as per
    AS4100 S5.11.5.2.

    Does not check for the presence of a stiffened end post.

    :param alpha_v: The stiffened web shear buckling parameter alpha_v as per
        AS4100 S5.11.5.2.
    :param d_p: The depth of the web panel.
    :param s: The length of the web or spacing of vertical stiffeners that meet the
        requirement of AS4100.
    """

    return 1 + ((1 - alpha_v) / (1.15 * alpha_v * (1 + (s / d_p) ** 2) ** 0.5))


def as_s5_15_3(*, gamma, a_w, alpha_v, v_star, v_u, d_p, s, phi=0.9):
    """
    Calculate the minimum area of a transverse shear stiffener as per AS4100 S5.15.3

    :param gamma: The stiffener type factor.
    :param a_w: The area of the web.
    :param alpha_v: The stiffened web shear buckling parameter alpha_v as per
        AS4100 S5.11.5.2.
    :param v_star: The shear in the web.
    :param v_u: The ultimate shear capacity of the web.
    :param d_p: The depth of the web panel.
    :param s: The length of the web or spacing of vertical stiffeners that meet the
        requirement of AS4100.
    :param phi: The capacity reduction factor
    """

    a_1 = 0.5 * gamma * a_w
    a_2 = 1 - alpha_v
    a_3 = v_star / (phi * v_u)
    a_4 = (s / d_p) - ((s / d_p) ** 2 / (1 + (s / d_p) ** 2) ** 0.5)

    return a_1 * a_2 * a_3 * a_4


def v_by(*, d_f, t_p, f_up):
    """
    Calculate the bolt hole bearing capacity, as limited by local yielding, as per
    AS4100 S9.2.2.4

    :param d_f: The fastener diameter.
    :param t_p: The thickness of the plate.
    :param f_up: The ultimate tensile stress of the plate.
    """

    return 3.2 * d_f * t_p * f_up


def v_bt(*, a_e, t_p, f_up):
    """
    Calculate the bolt hole bearing capacity, as limited by tear-out, as per
    AS4100 S9.2.2.4

    :param a_e: Fastener edge distance.
    :param t_p: Thickness of plate.
    :param f_up: The ultimate tensile stress of the plate.
    """

    return a_e * t_p * f_up


def v_w(size, f_uw, k_r=1.0, phi_weld=0.8):
    """
    Calculate the capacity of a fillet weld as per AS4100-2020

    :param size: The leg length of the weld.
        Future versions of this function may be extended to
        allow for uneven legs.
    :param f_uw: The ultimate strength of the weld metal.
    :param k_r: The weld length parameter. For welds less than
        1.7m long (almost all welds) this is 1.00.
    :param phi_weld: The capacity reduction factor.
    """

    t_t = size / (2**0.5)
    return phi_weld * 0.60 * t_t * k_r * f_uw
