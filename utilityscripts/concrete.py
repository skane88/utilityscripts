"""
File to contain some utilities for working with concrete.
"""

import re
from math import pi

from humre import (
    DIGIT,
    chars,
    exactly,
    group,
    one_or_more,
    one_or_more_group,
    zero_or_more,
)

MESH_DATA = {
    "RL1218": {"bar_dia": 11.9, "pitch": 100, "cross_bar_dia": 7.6, "cross_pitch": 200},
    "RL1018": {"bar_dia": 9.5, "pitch": 100, "cross_bar_dia": 7.6, "cross_pitch": 200},
    "RL818": {"bar_dia": 7.6, "pitch": 100, "cross_bar_dia": 7.6, "cross_pitch": 200},
    "SL102": {"bar_dia": 9.5, "pitch": 200, "cross_bar_dia": 9.5, "cross_pitch": 200},
    "SL92": {"bar_dia": 8.6, "pitch": 200, "cross_bar_dia": 8.6, "cross_pitch": 200},
    "SL82": {"bar_dia": 7.6, "pitch": 200, "cross_bar_dia": 7.6, "cross_pitch": 200},
    "SL72": {"bar_dia": 6.75, "pitch": 200, "cross_bar_dia": 6.75, "cross_pitch": 200},
    "SL62": {"bar_dia": 6.0, "pitch": 200, "cross_bar_dia": 6.0, "cross_pitch": 200},
    "SL81": {"bar_dia": 7.6, "pitch": 100, "cross_bar_dia": 7.6, "cross_pitch": 200},
}

D500N_STRESS_STRAIN = [[-0.05, -0.0025, 0, 0.0025, 0.05], [-500, -500, 0, 500, 500]]
D500L_STRESS_STRAIN = [[-0.015, -0.0025, 0, 0.0025, 0.015], [-500, -500, 0, 500, 500]]
R250N_STRESS_STRAIN = [[-0.05, -0.0025, 0, 0.0025, 0.05], [-500, -500, 0, 500, 500]]


def circle_area(dia):

    return pi * (dia**2) / 4


def reo_area(
    bar_spec: str = None, width: float = 1000, main_direction: bool = True
) -> float:
    """
    Calculate areas of reinforcement from a standard Australian specification code.
    """

    bar_pattern = exactly(
        1,
        group(chars(*("L", "N", "R", "Y")) + one_or_more(DIGIT)),
    )

    no_bars = re.compile(
        (zero_or_more(group(f"{one_or_more_group(DIGIT)}(-)")) + bar_pattern)
    )

    bars_with_spacing = re.compile(
        bar_pattern
        + exactly(
            1, group(exactly(1, group(chars(*("-", "@")))) + group(one_or_more(DIGIT)))
        )
    )

    mesh = re.compile(
        exactly(
            1,
            group(
                exactly(1, group(chars(*("S", "R")) + "L"))
                + exactly(1, group(one_or_more(DIGIT)))
            ),
        )
    )

    is_no_bars = no_bars.fullmatch(bar_spec)
    is_bars_spacing = bars_with_spacing.fullmatch(bar_spec)
    is_mesh = mesh.fullmatch(bar_spec)

    all_matches = [is_no_bars, is_bars_spacing, is_mesh]

    if all(x is not None for x in all_matches):
        raise ValueError(
            "Expected bar specification to match only one regular expression."
        )

    if all(x is None for x in all_matches):
        raise ValueError(
            "Expected designation to match one of the following bar designations:\n"
            + "number-bar\n"
            + "bar-spacing or \n"
            + "mesh"
        )

    if is_no_bars:

        no_bars = is_no_bars[2]

        no_bars = 1 if no_bars is None else int(no_bars)
        bar_type = is_no_bars[4]
        bar_dia = int(bar_type[1:])

    if is_bars_spacing:

        bar_type = is_bars_spacing[1]
        bar_dia = int(bar_type[1:])
        bar_spacing = is_bars_spacing[4]
        no_bars = width / int(bar_spacing)

    if is_mesh:

        mesh_data = MESH_DATA[bar_spec]

        pitch = mesh_data["pitch"]
        cross_pitch = mesh_data["cross_pitch"]

        if main_direction:
            bar_dia = mesh_data["bar_dia"]
            bar_spacing = pitch
        else:
            bar_dia = mesh_data["cross_bar_dia"]
            bar_spacing = cross_pitch

        no_bars = width / int(bar_spacing)

    bar_area = 0.25 * pi * bar_dia**2

    return bar_area * no_bars


def alpha_2(f_c):

    return max(0.67, 0.85 - 0.0015 * f_c)


def gamma(f_c):

    return max(0.67, 0.97 - 0.0025 * f_c)


def generate_rectilinear_block(f_c, strain, max_compression_strain: float = 0.003):
    """
    Generate a rectilinear stress-strain curve as required by AS3600.

    :param f_c: the characteristic compressive strength of the concrete.
    :param max_compression_strain: the maximum allowable compressive strain in the
        concrete. According to AS3600 S8.1.3 this is 0.003.
    """

    if strain > max_compression_strain:
        return 0

    gamma_val = gamma(f_c)

    if strain < max_compression_strain - max_compression_strain * gamma_val:
        return 0

    return alpha_2(f_c) * f_c


def M_uo_min(Z, f_ct_f, P_e=0, A_g=0, e=0):
    """
    Calculate the minimum required moment capacity.

    :param Z: the uncracked section modulus, taken at the face of the section at which
        cracking occurs.
    :param f_ct_f: the characteristic flexural strength of the concrete.
    :param P_e: effective prestress force, accounting for losses.
    :param A_g: gross area.
    :param e: prestress eccentricity from the centroid of the uncracked section.
    """

    return 1.2 * (Z * (f_ct_f + P_e / A_g) + P_e * e)


def M_uo():
    """
    Method to calculate the concrete capacity of a rectangular beam as per AS3600.
    """

    pass


def V_uo(f_c, u, d_om, beta_h=1.0):
    """
    Calculate the punching shear capacity.

    :param f_c: The characteristic strength of the concrete. Units are MPa.
    :param u: The punching shear perimeter. Units are m.
    :param d_om: The effective depth in punching shear. Units are m.
    :param beta_h: The shape factor for the punching shear area.
    :returns: Punching shear capacity in kN, provided input units are as specified.
    """

    def f_cv(f_c, beta_h=1):
        """
        Helper method to determine the punching shear strength.

        :param f_c: The characteristic strength of the concrete.
        :param beta_h: The shape factor for the punching shear area.
        """

        f_cv_1 = 0.17 * (1 + 2 / beta_h)
        f_cv_2 = 0.34

        return min(f_cv_1, f_cv_2) * f_c**0.5

    return f_cv(f_c, beta_h) * u * d_om * 1000  # *1000 to convert to kN.
