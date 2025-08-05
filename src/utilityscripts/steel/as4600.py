"""
Python file to determine steel design properties based on AS4600
"""

from enum import StrEnum
from math import pi

import numpy as np


class S5342ConnType(StrEnum):
    SINGLE_SHEAR_BOTH_WASHERS = "single shear both washers"
    SINGLE_SHEAR_ONE_WASHER = "single shear one washer"
    SINGLE_SHEAR_OVERSIZED_HOLES = "single shear oversized holes"
    SINGLE_SHEAR_OVERSIZED_HOLES_PERPENDICULAR = (
        "single shear oversized holes perpendicular"
    )
    DOUBLE_SHEAR_INNER = "double shear inner"
    DOUBLE_SHEAR_INNER_OVERSIZED_HOLES = "double shear inner oversized holes"
    DOUBLE_SHEAR_INNER_OVERSIZED_HOLES_PERPENDICULAR = (
        "double shear inner oversized holes perpendicular"
    )


def f_cr_plate(*, k, elastic_modulus, thickness, width, poisson_ratio=0.3):
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


def slenderness_ratio(*, f_star, f_cr):
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


def rho(*, slenderness):
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


def s5_3_2_v_f_phi(*, f_u, f_y):
    """
    Calculate the partial safety factor for the tearout capacity of a bolt hole,
    as per AS4600 S5.3.2

    Parameters
    ----------
    f_u : float
        The ultimate tensile strength of the plate.
    f_y : float
        The yield strength of the plate.
    """

    if f_u / f_y >= 1.05:  # noqa: PLR2004
        return 0.70

    return 0.60


def s5_3_2_v_f(*, t, e, f_u):
    """
    Calculate the tearout capacity of a bolt hole, ignoring deformation,
    as per AS4600 S5.3.2

    Parameters
    ----------
    t : float
        The thickness of the plate.
    e : float
        The edge distance.
    f_u : float
        The ultimate tensile strength of the plate.
    """

    return t * e * f_u


def s5_3_3_n_f(*, d_f, s_f, a_n, f_u):
    """
    Calculate the net tension capacity of the section at bolt holes,
    as per AS4600 S5.3.3

    Parameters
    ----------
    d_f : float
        The nominal fastener diameter.
    s_f : float
        The spacing between bolt holes, perpendicular to the direction of loading.
        If a single bolt hole, use the width of the sheet.
    a_n : float
        The net area of the section.
    f_u : float
        The ultimate tensile strength of the plate.
    """

    return (0.9 + (0.1 * d_f / s_f)) * a_n * f_u


def s5_3_4_2_v_b(*, alpha, c, d_f, t, f_u):
    """
    Calculate the bearing capacity of a bolt hole, ignoring deformation,
    as per AS4600 S5.4.3.2

    Parameters
    ----------
    alpha : float
        A modification factor for the type of bearing connection.
    c : float
        The bearing factor given in table 5.3.4.2,
    d_f : float
        The nominal fastener diameter.
    t : float
        The thickness of the plate.
    f_u : float
        The ultimate tensile strength of the plate.
    """

    return alpha * c * d_f * t * f_u


def t5_3_4_2_alpha(conn_type: S5342ConnType) -> float:
    """
    Calculate the modification factor (alpha) for connection beearing type as per
    AS4600 Table 5.3.4.2(a)

    Parameters
    ----------
    conn_type : S5342ConnType
        The connection type.
    """

    data = {
        S5342ConnType.SINGLE_SHEAR_BOTH_WASHERS: 1.00,
        S5342ConnType.SINGLE_SHEAR_ONE_WASHER: 0.75,
        S5342ConnType.SINGLE_SHEAR_OVERSIZED_HOLES: 0.70,
        S5342ConnType.SINGLE_SHEAR_OVERSIZED_HOLES_PERPENDICULAR: 0.55,
        S5342ConnType.DOUBLE_SHEAR_INNER: 1.33,
        S5342ConnType.DOUBLE_SHEAR_INNER_OVERSIZED_HOLES: 1.10,
        S5342ConnType.DOUBLE_SHEAR_INNER_OVERSIZED_HOLES_PERPENDICULAR: 0.90,
    }

    return data[conn_type]


def t5_3_4_2_c(*, d_f, t):
    """
    Calculate the bearing factor for a bolt hole as per AS4600 Table 5.3.4.2

    Parameters
    ----------
    d_f : float
        The nominal fastener diameter. In m.
    t : float
        The thickness of the plate. In m.
    """

    if not (0.00042 <= t <= 0.00476):  # noqa: PLR2004
        raise ValueError(f"t must be between 0.42 and 4.76. {t=:.2f}")

    if d_f / t < 10:  # noqa: PLR2004
        return 3.0

    if d_f / t <= 22:  # noqa: PLR2004
        return 4 - 0.1 * (d_f / t)

    return 1.8


def s5_3_4_3(*, t, d_f, f_u):
    """
    Calculate the bearing capacity at a nominal 6mm deflection.

    Parameters
    ----------
    t : float
        The thickness of the plate, in m.
    d_f : float
        The nominal fastener diameter, in m.
    f_u : float
        The ultimate strength of the plate, in MPa.

    Returns
    -------
    float
        The capacity in kN
    """

    return (0.183 * t * 1000 + 1.53) * d_f * t * f_u * 1000


def s5_4_2_3_n_t(*, d_f, s_f, a_n, f_u, no_rows):
    """
    Calculate the net tension capacity of a screwed connection,
    as per AS4600 S5.4.2.3.

    Parameters
    ----------
    d_f : float
        The nominal fastener diameter.
    s_f : float
        The spacing between bolt holes, perpendicular to the direction of loading.
        If a single bolt hole, use the width of the sheet.
    a_n : float
        The net area of the section.
    f_u : float
        The ultimate tensile strength of the plate.
    no_rows : int
        The number of rows of bolt holes in the direction of loading.
    """

    n_t_net = a_n * f_u

    if no_rows != 1:
        return n_t_net

    return min((2.5 * d_f / s_f) * n_t_net, n_t_net)


def s5_4_2_4_v_b(*, d_f, t_1, t_2, f_u_1, f_u_2):
    """
    Calculate the bearing capacity of a screwed connection,
    as per AS4600 S5.4.2.4.

    Parameters
    ----------
    d_f : float
        The nominal fastener diameter.
    t_1 : float
        The thickness of the part in contact with the screw head.
    t_2 : float
        The thickness of the part not in contact with the screw head.
    f_u_1 : float
        The ultimate tensile strength of the part in contact with the screw head.
    f_u_2 : float
        The ultimate tensile strength of the part not in contact with the screw head.
    """

    c_1 = t5_4_2_4_c(d_f=d_f, t=t_1)
    c_2 = t5_4_2_4_c(d_f=d_f, t=t_2)

    v_b_1 = c_1 * t_1 * d_f * f_u_1
    v_b_2 = c_2 * t_2 * d_f * f_u_2
    v_b_3 = 4.2 * (t_2**3 * d_f) ** 0.5 * f_u_2

    t_ratio = t_2 / t_1

    ratio = np.asarray([1.0, 2.5])
    vals = np.asarray([min(v_b_1, v_b_2, v_b_3), min(v_b_1, v_b_2)])

    return np.interp(t_ratio, ratio, vals)


def t5_4_2_4_c(*, d_f, t):
    """
    Calculate the bearing factor for a screw hole, as per AS4600 S5.4.2.4
    """

    d_f_t = d_f / t

    if d_f_t < 6:  # noqa: PLR2004
        return 2.7

    if d_f_t <= 13.0:  # noqa: PLR2004
        return 3.3 - 0.1 * d_f_t

    return 2.0


def s5_4_2_5_v_fv_phi(*, f_u, f_y):
    """
    Calculate the partial safety factor for the tearout capacity of a
    screwed connection, as per AS4600 S5.4.2.5

    Parameters
    ----------
    f_u : float
        The ultimate tensile strength of the plate.
    f_y : float
        The yield strength of the plate.
    """

    if f_u / f_y >= 1.05:  # noqa: PLR2004
        return 0.70

    return 0.60


def s5_4_2_5_v_fv(*, t, e, f_u):
    """
    Calculate the tearout capacity of a screwed connection,
    as per AS4600 S5.4.2.5

    Parameters
    ----------
    t : float
        The thickness of the plate.
    e : float
        The edge distance.
    f_u : float
        The ultimate tensile strength of the plate.
    """

    return t * e * f_u
