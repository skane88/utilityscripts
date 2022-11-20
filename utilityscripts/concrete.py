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
    "RL1118": {
        "bar_dia": 10.65,
        "pitch": 100,
        "cross_bar_dia": 7.6,
        "cross_pitch": 200,
    },
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


# Regular expressions for bar and mesh specifications.
BAR_RE = exactly(
    1,
    group(chars(*("L", "N", "R", "Y")) + one_or_more(DIGIT)),
)
MESH_RE = exactly(
    1,
    group(
        exactly(1, group(chars(*("S", "R")) + "L"))
        + exactly(1, group(one_or_more(DIGIT)))
    ),
)


def circle_area(dia):

    return pi * (dia**2) / 4


def is_bar(bar_spec: str) -> bool:
    """
    Determine if a bar specification matches a standard bar code.
    """

    return reo_properties(bar_spec)["is_bar"]


def is_mesh(bar_spec: str) -> bool:
    """
    Determine if a bar specification matches a standard mesh code.
    """

    return reo_properties(bar_spec)["is_mesh"]


def reo_properties(bar_spec: str):
    """
    Returns a dictionary with a number of properties from a standard bar specification.
    """

    ret_val = {
        "is_bar": None,
        "is_mesh": None,
        "bar_type": None,
        "main_dia": None,
        "secondary_dia": None,
        "main_spacing": None,
        "secondary_spacing": None,
        "no_main": None,
        "no_secondary": None,
    }

    no_bars = re.compile(
        (zero_or_more(group(f"{one_or_more_group(DIGIT)}(-)")) + BAR_RE)
    )

    bars_with_spacing = re.compile(
        BAR_RE
        + exactly(
            1, group(exactly(1, group(chars(*("-", "@")))) + group(one_or_more(DIGIT)))
        )
    )

    is_no_bars = no_bars.fullmatch(bar_spec)
    is_bars_spacing = bars_with_spacing.fullmatch(bar_spec)
    mesh = re.compile(MESH_RE).fullmatch(bar_spec)

    all_matches = [is_no_bars, is_bars_spacing, mesh]

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
        no_bars = 1 if is_no_bars[1] is None else is_no_bars[1][:-1]
        no_bars = 1 if no_bars is None else int(no_bars)
        bar_type = is_no_bars[4][:1]
        bar_dia = int(is_no_bars[4][1:])

        ret_val["is_bar"] = True
        ret_val["is_mesh"] = False
        ret_val["main_dia"] = bar_dia
        ret_val["bar_type"] = bar_type
        ret_val["main_spacing"] = None
        ret_val["secondary_dia"] = None
        ret_val["secondary_spacing"] = None
        ret_val["no_main"] = no_bars
        ret_val["no_secondary"] = None

    if is_bars_spacing:
        bar_type = is_bars_spacing[1][:1]
        bar_dia = int(is_bars_spacing[1][1:])
        bar_spacing = is_bars_spacing[4]

        ret_val["is_bar"] = True
        ret_val["is_mesh"] = False
        ret_val["main_dia"] = bar_dia
        ret_val["bar_type"] = bar_type
        ret_val["main_spacing"] = float(bar_spacing)
        ret_val["secondary_dia"] = None
        ret_val["secondary_spacing"] = None
        ret_val["no_main"] = None
        ret_val["no_secondary"] = None

    if mesh:

        mesh_data = MESH_DATA[bar_spec]

        pitch = mesh_data["pitch"]
        cross_pitch = mesh_data["cross_pitch"]

        bar_dia = mesh_data["bar_dia"]
        bar_spacing = pitch
        secondary_bar_dia = mesh_data["cross_bar_dia"]
        secondary_bar_spacing = cross_pitch

        ret_val["is_bar"] = True
        ret_val["is_mesh"] = False
        ret_val["main_dia"] = bar_dia
        ret_val["bar_type"] = bar_spec[:2]
        ret_val["main_spacing"] = bar_spacing
        ret_val["secondary_dia"] = secondary_bar_dia
        ret_val["secondary_spacing"] = secondary_bar_spacing
        ret_val["no_main"] = None
        ret_val["no_secondary"] = None

    return ret_val


def reo_area(
    bar_spec: str,
    *,
    width: float = 1000,
    main_direction: bool = True,
):
    """
    Calculate areas of reinforcement from a standard Australian specification code.

    For specifications based on bar spacing (e.g. "N16-200") or for a mesh (e.g. "SL82",
    a width must be provided to calculate the area correctly, otherwise the value
    returned is over a nominal 1000mm width of reinforcement.

    :param bar_spec: a standard Australian bar specification code.
    :param main_direction: for mesh, calculate the area in the main or secondary
        direction?
    :param width: the width over which to calculate the area.
    """

    reo_data = reo_properties(bar_spec=bar_spec)

    if main_direction:
        bar_dia = reo_data["main_dia"]
        no_bars = reo_data["no_main"]
        bar_spacing = reo_data["main_spacing"]
    else:
        bar_dia = reo_data["secondary_dia"]
        no_bars = reo_data["no_secondary"]
        bar_spacing = reo_data["secondary_spacing"]

    if no_bars is None:
        no_bars = width / bar_spacing

    single_bar_area = 0.25 * pi * bar_dia**2
    total_area = single_bar_area * no_bars
    area_unit_width = total_area / (width / 1000)

    return {
        "single_bar_area": single_bar_area,
        "total_area": total_area,
        "area_unit_width": area_unit_width,
        "no_bars": no_bars,
        "width": width,
    }


def alpha_2(f_c):

    return max(0.67, 0.85 - 0.0015 * f_c)


def gamma(f_c):

    return max(0.67, 0.97 - 0.0025 * f_c)


def generate_rectilinear_block(f_c, max_compression_strain: float = 0.003):
    """
    Generate a point on a rectilinear stress-strain curve as required by AS3600.

    :param f_c: the characteristic compressive strength of the concrete.
    :param max_compression_strain: the maximum allowable compressive strain in the
        concrete. According to AS3600 S8.1.3 this is 0.003.
    """

    gamma_val = gamma(f_c)
    min_strain = max_compression_strain * (1 - gamma_val)

    compressive_strength = alpha_2(f_c) * f_c

    return [
        [0, min_strain, max_compression_strain],
        [0, compressive_strength, compressive_strength],
    ]


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
