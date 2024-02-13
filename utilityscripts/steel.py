"""
To contain some utilities for steel design
"""
from math import pi


def alpha_m(*, M_m, M_2, M_3, M_4):
    """
    Determines the moment modification factor as per AS4100 S5.6.1.1.a.iii

    :param M_m: The maximum moment.
    :param M_2: The moment at the 1st 1/4 point.
    :param M_3: The moment at mid-span.
    :param M_4: The moment at the 2nd 1/4 point.
    """

    return 1.7 * M_m / (M_2**2 + M_3**2 + M_4**2) ** 0.5


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


def jack_bolt_effort(*, load, p, r_effort, r_bolt, mu, against: bool = True):
    """
    Calculate the force required to turn a jacking bolt.

    :param load: The weight or load.
    :param p: The pitch of the screw
    :param r_effort: The radius of the effort force.
    :param r_bolt: The radius of the screw / bolt.
    :param mu: The assumed coefficient of friction.
        Note that for bolts this can vary dramatically depending on thread condition.
    :param against: Is the jacking against the load or with the load?
        Jacking against the load (e.g. raising a load under gravity)
        requires larger force than jacking with the load (e.g. lowering a load under gravity).
    """

    if against:
        return (
            load
            * ((2 * pi * mu * r_bolt + p) / (2 * pi * r_bolt - mu * p))
            * (r_bolt / r_effort)
        )

    return (
        load
        * ((2 * pi * mu * r_bolt - p) / (2 * pi * r_bolt + mu * p))
        * (r_bolt / r_effort)
    )


def local_thickness_reqd(
    *, point_force, f_y, phi=0.9, width_lever: tuple[float, float] = None
):
    """
    Calculate the local thickness of a plate required to transfer a load back to a support by local bending.
    This assumes a 45deg load dispersion through the plate towards the support, ignoring and load transfer into
    the 2nd dimension.

    :param point_force: The load to transfer.
    :param f_y: The yield stress of the plate.
    :param phi: A capacity reduction factor.
    :param width_lever: A tuple containing the width of the applied load and the lever arm between load and support.
        If None, the load is treated as a point load.
    """

    if width_lever is None:
        return ((2 * point_force) / (phi * f_y)) ** 0.5

    lw = width_lever[0]
    lever = width_lever[1]

    if lever == 0:
        width_lever_ratio = 0
    else:
        width_lever_ratio = lw / lever

    return 2 / (((phi * f_y / point_force) * (width_lever_ratio + 2)) ** 0.5)
