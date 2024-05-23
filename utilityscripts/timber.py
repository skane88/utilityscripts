"""
Python module to contain functions for working with timber.
"""

from math import pi
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_PATH = Path(Path(__file__).parent.parent) / Path("data")


def get_k_1(
    t_s=0,
    *,
    t_min: float | None = None,
    t_hour: float | None = None,
    t_day: float | None = None,
    t_month: float | None = None,
    t_year: float | None = None,
    member: bool = True,
):
    """
    Get the k_1 factor for design.

    :param t_s: The no. of seconds of load duration.
    :param t_min: The no. of minutes of load duration. Optional.
    :param t_hour: The no. of hours of load duration. Optional.
    :param t_day: The no. of days of load duration. Optional.
    :param t_month: The no. of months of load duration. Optional.
    :param t_year: The no. of years of load duration. Optional.
    :param member: Should k_1 be for the member or the joint.
    """

    if t_min is not None:
        t_s += t_min * 60
    if t_hour is not None:
        t_s += t_hour * 60 * 60
    if t_day is not None:
        t_s += t_day * 24 * 60 * 60
    if t_month is not None:
        t_s += t_month * (365 / 12) * 24 * 60 * 60
    if t_year is not None:
        t_s += t_year * 365 * 24 * 60 * 60

    if t_s < 0:
        raise ValueError(f"Time should be > 0. Time = {t_s:.1f}")

    k_1_df = pd.read_excel(_DATA_PATH / Path("timber_data.xlsx"), sheet_name="k_1")

    x_t = k_1_df.load_duration_s.to_numpy()
    y_k_1 = k_1_df.k_1_member.to_numpy() if member else k_1_df.k_1_joint.to_numpy()

    return np.interp(t_s, x_t, y_k_1)


def get_k_4_unseasoned(dim: float):
    """
    Return the k_4 factor for design.

    :param dim: The minimum dimension of the timber. In m.
    """

    if dim <= 0:
        raise ValueError(f"Dimension should be greater than 0. {dim=:.1f}")

    k_4_df = pd.read_excel(_DATA_PATH / Path("timber_data.xlsx"), sheet_name="k_4")

    x_d = k_4_df.min_dim
    y_k_4 = k_4_df.k_4

    return np.interp(dim, x_d, y_k_4)


def get_k_4_seasoned(emc: float):
    """
    Determine the moisture factor k_4 for seasoned timber.

    :param emc: The expected moisture content, EMC.
        Should be a float between 0 and 1, typ. in the range
        0.0 - 0.30 (e.g. 0% to 30% moisture content).
    """

    if emc <= 0.15:  # noqa: PLR2004
        return 1.0

    return max(0.7, 1 - 0.3 * (emc - 0.15) / 0.1)


def get_k_9(
    spacing: float = 0, length: float = 1.0, n_combined: float = 1, n_members: float = 1
):
    """
    Get the combination factor k_9 for design.

    :param spacing: The centre-centre spacing of parallel members in the
        load sharing system.
    :param length: The span length of the parallel members.
    :param n_combined: The number of members combined to form a single member.
    :param n_members: The number of parallel members in the load sharing system.
    """

    g_31_32_df = pd.read_excel(
        _DATA_PATH / Path("timber_data.xlsx"), sheet_name="g_31_32"
    )

    x_n = g_31_32_df.num
    y_g = g_31_32_df.g_31_32

    g_31 = np.interp(n_combined, x_n, y_g)
    g_32 = np.interp(n_members * n_combined, x_n, y_g)

    return max(g_31 + (g_32 - g_31) * (1 - 2 * spacing / length), 1.0)


def k_12_bending(rho_b, s_1):
    """
    Calculate the stability factor in bending.

    :param rho_b: The material constant for bending.
    :param s_1: The slenderness coefficient for beams.
    """

    rho_prod = rho_b * s_1

    if rho_prod <= 10.0:  # noqa: PLR2004
        return 1.0

    if rho_prod <= 20.0:  # noqa: PLR2004
        return 1.5 - 0.05 * rho_prod

    return 200 / (rho_prod**2)


def rho_b(elastic_modulus, f_b, load_ratio, *, seasoned: bool = True):
    """
    Calculate the material constant in bending:

    :param elastic_modulus: The elastic modulus of the timber.
    :param f_b: The characteristic strength in bending.
    :param load_ratio: What is the ratio of short term loads to long term loads.
    :param seasoned: Is the timber seasoned?
    """

    if seasoned:
        return 14.71 * ((elastic_modulus / f_b) ** -0.480) * (load_ratio**-0.061)

    return 11.63 * ((elastic_modulus / f_b) ** -0.435) * (load_ratio**-0.110)


def s_1_rect(
    *,
    d,
    b,
    l_a: float | str,
    load_on_c: bool = True,
    torsional_restraints: bool = False,
):
    """
    Calculate the parameter S1 for the simplifed case of rectangular beams.

    NOTE: If d < b, bending is about the minor axis, and S2 = 0.0 is returned.

    :param d: The depth of the section in the direction of bending.
    :param b: The width of the section.
    :param l_a: The distance between restraints.
        If continuously restrained, provide the str "clr".
        If torsional_restraints == False, this is L_ay.
        If torsional_restraints == True, this is L_aφ
    :param load_on_c: Is the load on the compression edge?
    :param torsional_restraints: Are there torsional restraints?
        Torsional restraints are only valid if load_on_c == False.
    :raises: Raises an error if torsional_restraints == True and
        load_on_c == True.
    """

    if d < b:
        # bending is actually about the minor axis.
        return 0.0

    if isinstance(l_a, str):
        if l_a != "clr":
            raise ValueError(
                "The only valid string is 'clr' for continuous lateral restraints."
                + " Otherwise provide a number."
                + f" Received: {l_a=}"
            )

        if torsional_restraints:
            raise ValueError(
                "No case where torsional restrains are considered"
                + " combined with Continuous Lateral Restraints."
            )

        if load_on_c:
            return 0.0

        return 2.25 * d / b

    if load_on_c:
        if torsional_restraints:
            raise ValueError(
                "No case where torsional restraints are considered"
                + " when load is on the compression edge."
            )

        return 1.25 * (d / b) * ((l_a / d) ** 0.5)

    if torsional_restraints:
        return (1.5 * (d / b)) / ((((pi * d) / l_a) ** 2 + 0.4) ** 0.5)

    return ((d / b) ** 1.35) * ((l_a / d) ** 0.25)


def m_d(*, phi, k_1, k_4, k_9, k_12, f_b, z, k_6=0.9):
    """
    Determine the capacity of a timber beam in bending.

    :param phi: The capacity reduction factor.
    :param k_1: Load duration factor.
    :param k_4: Seasoning / moisture content factor.
    :param k_6: Temperature factor.
        Defaults to the minimum value for QLD north of 25deg
        or rest of Aus north of 15deg.
    :param k_9: Strength sharing factor.
    :param k_12: Stability factor.
    :param f_b: Characteristic bending strength of the timber.
    :param z: Elastic section modulus.
    """

    return phi * k_1 * k_4 * k_6 * k_9 * k_12 * f_b * z
