"""
File to contain some utilities for working with concrete.
"""

import re
from collections import namedtuple
from math import pi

from humre import (
    DIGIT,
    chars,
    exactly,
    group,
    group_either,
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
    "F1218": {"bar_dia": 12.5, "pitch": 100, "cross_bar_dia": 8.0, "cross_pitch": 200},
    "F1118": {"bar_dia": 11.2, "pitch": 100, "cross_bar_dia": 8.0, "cross_pitch": 200},
    "F1018": {"bar_dia": 10.0, "pitch": 100, "cross_bar_dia": 8.0, "cross_pitch": 200},
    "F918": {"bar_dia": 9.0, "pitch": 100, "cross_bar_dia": 8.0, "cross_pitch": 200},
    "F818": {"bar_dia": 8.0, "pitch": 100, "cross_bar_dia": 8.0, "cross_pitch": 200},
    "F718": {"bar_dia": 7.1, "pitch": 100, "cross_bar_dia": 8.0, "cross_pitch": 200},
    "F928": {"bar_dia": 9.0, "pitch": 200, "cross_bar_dia": 8.0, "cross_pitch": 250},
    "F828": {"bar_dia": 8.0, "pitch": 200, "cross_bar_dia": 8.0, "cross_pitch": 250},
    "F81": {"bar_dia": 8.0, "pitch": 100, "cross_bar_dia": 8.0, "cross_pitch": 100},
    "F102": {"bar_dia": 10.0, "pitch": 200, "cross_bar_dia": 10.0, "cross_pitch": 200},
    "F92": {"bar_dia": 9.0, "pitch": 200, "cross_bar_dia": 9.0, "cross_pitch": 200},
    "F82": {"bar_dia": 8.0, "pitch": 200, "cross_bar_dia": 8.0, "cross_pitch": 200},
    "F72": {"bar_dia": 7.1, "pitch": 200, "cross_bar_dia": 7.1, "cross_pitch": 200},
    "F62": {"bar_dia": 6.3, "pitch": 200, "cross_bar_dia": 6.3, "cross_pitch": 200},
    "F52": {"bar_dia": 5.0, "pitch": 200, "cross_bar_dia": 5.0, "cross_pitch": 200},
    "F42": {"bar_dia": 4.0, "pitch": 200, "cross_bar_dia": 4.0, "cross_pitch": 200},
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
        exactly(1, group_either(chars(*("S", "R")) + "L", "F"))
        + exactly(1, group(one_or_more(DIGIT)))
    ),
)


def circle_area(diameter):
    """
    Calculate the area of a circle.

    :param dia: the diameter of the circle.
    """

    return pi * (diameter**2) / 4


def is_bar(bar_spec: str) -> bool:
    """
    Determine if a bar specification matches a standard bar code.
    """

    return reo_properties(bar_spec).is_bar


def is_mesh(bar_spec: str) -> bool:
    """
    Determine if a bar specification matches a standard mesh code.
    """

    return reo_properties(bar_spec).is_mesh


def reo_properties(bar_spec: str, *, main_width=1000.0, secondary_width=1000.0):
    """
    Returns a dictionary with a number of properties from a standard bar specification.

    :param bar_spec: a standard Australian bar specification code.
    :param main_width: the width over which to calculate the area of the main bars.
    :param secondary_width:
        the width over which to calculate the area of the secondary bars.
    """

    ReoProperties = namedtuple(
        "ReoProperties",
        [
            "is_bar",
            "is_mesh",
            "bar_type",
            "main_dia",
            "secondary_dia",
            "main_spacing",
            "secondary_spacing",
            "no_main",
            "no_secondary",
            "main_bar_area",
            "secondary_bar_area",
            "main_area_total",
            "secondary_area_total",
            "main_area_unit",
            "secondary_area_unit",
            "main_width",
            "secondary_width",
        ],
    )

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
        bar_type = is_no_bars[4][:1]
        main_dia = int(is_no_bars[4][1:])

        main_bar_area = 0.25 * pi * main_dia**2

        no_main = 1 if is_no_bars[1] is None else is_no_bars[1][:-1]
        no_main = 1 if no_bars is None else int(no_main)

        main_area_total = main_bar_area * no_main

        ret_val = ReoProperties(
            is_bar=True,
            is_mesh=False,
            main_dia=main_dia,
            bar_type=bar_type,
            main_spacing=None,
            secondary_dia=None,
            secondary_spacing=None,
            no_main=no_main,
            no_secondary=None,
            main_bar_area=main_bar_area,
            secondary_bar_area=None,
            main_area_total=main_area_total,
            secondary_area_total=None,
            main_area_unit=main_area_total / (main_width / 1000),
            secondary_area_unit=None,
            main_width=main_width,
            secondary_width=None,
        )

    if is_bars_spacing:
        bar_type = is_bars_spacing[1][:1]
        main_dia = int(is_bars_spacing[1][1:])
        bar_spacing = float(is_bars_spacing[4])

        no_main = main_width / bar_spacing
        main_bar_area = 0.25 * pi * main_dia**2
        main_area_total = main_bar_area * no_main

        ret_val = ReoProperties(
            is_bar=True,
            is_mesh=False,
            main_dia=main_dia,
            bar_type=bar_type,
            main_spacing=bar_spacing,
            secondary_dia=None,
            secondary_spacing=None,
            no_main=no_main,
            no_secondary=None,
            main_bar_area=main_bar_area,
            secondary_bar_area=None,
            main_area_total=main_area_total,
            secondary_area_total=None,
            main_area_unit=main_area_total / (main_width / 1000),
            secondary_area_unit=None,
            main_width=main_width,
            secondary_width=None,
        )

    if mesh:
        mesh_data = MESH_DATA[bar_spec]

        main_dia = mesh_data["bar_dia"]
        pitch = mesh_data["pitch"]
        no_main = main_width / pitch

        secondary_dia = mesh_data["cross_bar_dia"]
        cross_pitch = mesh_data["cross_pitch"]
        no_secondary = secondary_width / cross_pitch

        main_bar_area = 0.25 * pi * main_dia**2
        secondary_bar_area = 0.25 * pi * secondary_dia**2

        main_area_total = main_bar_area * no_main
        secondary_area_total = secondary_bar_area * no_secondary

        ret_val = ReoProperties(
            is_bar=False,
            is_mesh=True,
            main_dia=main_dia,
            bar_type=mesh[2],
            main_spacing=pitch,
            secondary_dia=secondary_dia,
            secondary_spacing=cross_pitch,
            no_main=no_main,
            no_secondary=no_secondary,
            main_bar_area=main_bar_area,
            secondary_bar_area=secondary_bar_area,
            main_area_total=main_area_total,
            secondary_area_total=secondary_area_total,
            main_area_unit=main_area_total / (main_width / 1000),
            secondary_area_unit=secondary_area_total / (secondary_width / 1000),
            main_width=main_width,
            secondary_width=secondary_width,
        )

    return ret_val


def alpha_2(f_c):
    """
    Calculate parameter alpha_2 as per AS3600-2018.

    :param f_c: The characteristic compressive strength of the concrete
    """

    return max(0.67, 0.85 - 0.0015 * f_c)


def gamma(f_c):
    """
    Calculate parameter gamma as per AS3600-2018.

    :param f_c: The characteristic compressive strength of the concrete.
    """

    return max(0.67, 0.97 - 0.0025 * f_c)


def generate_rectilinear_block(*, f_c, max_compression_strain: float = 0.003):
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


def m_uo_min(*, z, f_ct_f, p_e=0, a_g=0, e=0):
    """
    Calculate the minimum required moment capacity.

    :param z: the uncracked section modulus, taken at the face of the section at which
        cracking occurs.
    :param f_ct_f: the characteristic flexural strength of the concrete.
    :param p_e: effective prestress force, accounting for losses.
    :param a_g: gross area.
    :param e: prestress eccentricity from the centroid of the uncracked section.
    """

    return 1.2 * (z * (f_ct_f + p_e / a_g) + p_e * e)


def m_uo():
    """
    Method to calculate the concrete capacity of a rectangular beam as per AS3600.
    """

    pass


def v_uo(*, f_c, u, d_om, beta_h=1.0):
    """
    Calculate the punching shear capacity.

    :param f_c: The characteristic strength of the concrete. Units are MPa.
    :param u: The punching shear perimeter. Units are m.
    :param d_om: The effective depth in punching shear. Units are m.
    :param beta_h: The shape factor for the punching shear area.
    :returns: Punching shear capacity in kN, provided input units are as specified.
    """

    def f_cv(f_c, beta_h=1.0):
        """
        Helper method to determine the punching shear strength.

        :param f_c: The characteristic strength of the concrete.
        :param beta_h: The shape factor for the punching shear area.
        """

        f_cv_1 = 0.17 * (1 + 2 / beta_h)
        f_cv_2 = 0.34

        return min(f_cv_1, f_cv_2) * f_c**0.5

    return f_cv(f_c, beta_h) * u * d_om * 1000  # *1000 to convert to kN.


def c_d(*, bar_spacing, cover, bar_type: str = "straight", narrow_element: bool = True):
    """
    Calculate the cover & spacing parameter c_d as per AS3600 S13.1.2

    :param bar_spacing: The spacing between bars.
    :param cover: The cover to the bars.
        See figure 13.1.2.2 to determine which cover is appropriate.
    :param bar_type: The type of bar, either "straight", "hooked" or "looped"
    :param narrow_element: Is the element narrow (e.g. beam web, column)
        or wide (e.g. slab, inner bars of a band beam etc.)
    """

    if bar_type not in ["straight", "hooked", "looped"]:
        raise ValueError("Incorrect bar_type.")

    if narrow_element:
        if bar_type in ["straight", "hooked"]:
            return min(bar_spacing / 2, cover)
        return cover

    if bar_type == "straight":
        return min(bar_spacing / 2, cover)
    if bar_type == "hooked":
        return bar_spacing / 2
    return cover


def k_3(
    *, d_b, bar_spacing, cover, bar_type: str = "straight", narrow_element: bool = True
):
    """
    Calculate parameter k_3 as per AS3600 S13.1.2.2

    :param d_b: bar diamter.
        Units should be consistent with bar_spacing and cover.
    :param bar_spacing: The spacing between bars.
    :param cover: The cover to the bars.
        See figure 13.1.2.2 to determine which cover is appropriate.
    :param bar_type: The type of bar, either "straight", "hooked" or "looped"
    :param narrow_element: Is the element narrow (e.g. beam web, column)
        or wide (e.g. slab, inner bars of a band beam etc.)
    """

    c_d_calc = c_d(
        bar_spacing=bar_spacing,
        cover=cover,
        bar_type=bar_type,
        narrow_element=narrow_element,
    )

    k_3_calc = 1 - 0.15 * (c_d_calc - d_b) / d_b

    return max(
        min(
            k_3_calc,
            1.0,
        ),
        0.7,
    )


def l_syt(*, f_c, f_sy, d_b, k_1=1.0, k_3=1.0, k_4=1.0, k_5=1.0):
    """
    Calculate the development length of a bar as per AS3600 S13.1.2.3

    :param f_c: The concrete characteristic compressive strength, f'c. In MPa.
    :param f_sy: The steel yield strength (in MPa)
    :param d_b: The diameter of the bar (in mm)
    :param k_1: The member depth parameter.
    :param k_3: The bar spacing parameter.
    :param k_4: Transverse reinforcement parameter.
    :param k_5: Transverse stress parameter.
    :return: The development length in mm.
    """

    k_2 = (132 - d_b) / 100
    k_prod = max(0.7, k_3 * k_4 * k_5)
    l_calc = 0.5 * k_1 * k_prod * f_sy * d_b / (k_2 * f_c**0.5)
    l_min = 0.058 * f_sy * k_1 * d_b

    return max(l_calc, l_min)


class PadFooting:
    """
    Class to describe a basic pad footing.
    """

    def __init__(
        self,
        bx,
        bz,
        d_pad,
        bx_pedestal,
        bz_pedestal,
        h_pedestal,
        soil_level,
        washout_depth,
        concrete_density: float = 24.0,
    ):
        """

        :param bx: The pedestal width in the x direction.
        :param bz: The pedestal width in the z direction.
        :param d_pad: The depth of the footing pad.
        :param bx_pedestal: The pedestal width in the x direction.
        :param bz_pedestal: The pedestal width in the z direction.
        :param h_pedestal: The pedestal height.
        :param soil_level: The depth of soil above the top of the pad.
        :param washout_depth: The depth of soil to ignore for uplift assessment.
            This is typically an allowance for soil being washed out, but soil may
            also be removed for other reasons, such as excessive cleanup activities or
            operational changes around the footing.
        :param concrete_density: The density of concrete to use in the assessment.
            A default value of 24kN/m^3 is used, but if the density is different (due
            to reinforcement steel for example) or different units (e.g. 2400kg/m^3)
            are being used then update as required.
        """

        self.bx = bx
        self.bz = bz
        self.d_pad = d_pad
        self.bx_pedestal = bx_pedestal
        self.bz_pedestal = bz_pedestal
        self.h_pedestal = h_pedestal
        self.soil_level = soil_level
        self.washout_depth = washout_depth
        self.concrete_density = concrete_density

    @property
    def vol_concrete(self):
        """
        Return the volume of concrete in the footing.
        """

        return (
            self.bx * self.bz * self.d_pad
            + self.bx_pedestal * self.bz_pedestal * self.h_pedestal
        )

    @property
    def footing_mass(self):
        """
        The mass of the concrete in the footing.
        """

        return self.vol_concrete * self.concrete_density

    @property
    def soil_level_washout(self):
        """
        The depth of soil after accounting for washout.
        """

        return self.soil_level - self.washout_depth

    def vol_soil_vertical(self, *, washout: bool = True):
        """
        Return the volume of soil immediately above the footing.

        :param washout: Consider the effects of soil washout.
        """

        depth = self.soil_level_washout if washout else self.soil_level

        return (self.pad_area - self.bx_pedestal * self.bz_pedestal) * depth

    @property
    def pad_area(self):
        """
        Return the pad area.
        """

        return self.bx * self.bz

    @property
    def elastic_modulus_x(self):
        """
        Calculate the footing's elastic modulus in the x direction.

        Used to calculate pressures at serviceability.
        """

        return self.bz * self.bx**2 / 6

    @property
    def elastic_modulus_z(self):
        """
        Calculate the footing's elastic modulus in the z direction.

        Used to calculate pressures at serviceability.
        """

        return self.bx * self.bz**2 / 6
