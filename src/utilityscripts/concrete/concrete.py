"""
File to contain some utilities for working with concrete.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from math import pi

from humre import (  # type: ignore
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
    "SL81": {"bar_dia": 7.6, "pitch": 100, "cross_bar_dia": 7.6, "cross_pitch": 100},
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

# Regular expressions for bar and mesh specifications.
BAR_RE = exactly(
    1,
    group(chars(*("C", "L", "N", "R", "S", "W", "Y")) + one_or_more(DIGIT)),
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

    return reo_prop(bar_spec).is_bar


def is_mesh(bar_spec: str) -> bool:
    """
    Determine if a bar specification matches a standard mesh code.
    """

    return reo_prop(bar_spec).is_mesh


class BarSpec(StrEnum):
    C = "C"
    F = "F"
    L = "L"
    N = "N"
    R = "R"
    S = "S"
    W = "W"
    Y = "Y"
    RL = "RL"
    SL = "SL"


@dataclass(slots=True, kw_only=True)
class ReoProperties:
    bar_spec: str
    is_bar: bool
    is_mesh: bool
    bar_type: BarSpec
    main_dia: int
    secondary_dia: int | None
    main_spacing: float | None
    secondary_spacing: float | None
    no_main: int | float
    no_secondary: int | float | None
    main_width: float | None = None
    secondary_width: float | None = None

    @property
    def main_is_unit(self) -> bool:
        """
        Is the main area a unit area?
        """

        return self.no_main is None and self.main_width is None

    @property
    def secondary_is_unit(self) -> bool:
        """
        Is the secondary area a unit area?
        """

        return self.no_secondary is None and self.secondary_width is None

    @property
    def main_bar_area(self):
        return circle_area(self.main_dia)

    @property
    def main_area_total(self):
        """
        Calculate the total area of the main bars.
        If the total number of bars has not been provided and no width has been given,
        this is calculated based on a unit width.
        """

        if self.no_main is None:
            width = 1000 if self.main_width is None else self.main_width
            no_main = width / self.main_spacing

        else:
            no_main = self.no_main

        return self.main_bar_area * no_main

    @property
    def main_area_unit(self):
        return self.main_area_total / (self.main_width / 1000)

    @property
    def secondary_bar_area(self):
        if self.secondary_dia is None:
            return None
        return circle_area(self.secondary_dia)

    @property
    def secondary_area_total(self):
        """
        Calculate the total area of the secondary bars.
        If the total number of bars has not been provided and no width has been given,
        this is calculated based on a unit width.
        """

        if self.secondary_bar_area is None:
            return None

        if self.no_secondary is None:
            width = 1000 if self.secondary_width is None else self.secondary_width
            no_secondary = width / self.secondary_spacing

        else:
            no_secondary = self.no_secondary

        return self.secondary_bar_area * no_secondary

    @property
    def secondary_area_unit(self):
        if self.secondary_area_total is None or self.secondary_width is None:
            return None
        return self.secondary_area_total / (self.secondary_width / 1000)

    @property
    def area_mass(self):
        if self.main_width is None or self.secondary_width is None:
            return None

        return mesh_mass(
            main_diameter=self.main_dia,
            main_spacing=self.main_spacing,
            secondary_diameter=self.secondary_dia,
            secondary_spacing=self.secondary_spacing,
            main_width=self.main_width,
            secondary_width=self.secondary_width,
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}: {self.bar_spec}."
            + f" {'Mesh' if self.is_mesh else 'Bar'}"
            + f", bar type: {self.bar_type}.\n"
            + f"Main diameter: {self.main_dia}mm"
            + f", bar area: {self.main_bar_area:.1f}mm²"
            + f", total area: {self.main_area_total:.1f}mm² {' (unit)' if self.main_is_unit else ''}."
            + (
                (
                    f"\nSecondary Diameter: {self.secondary_dia}mm"
                    + f", bar area: {self.secondary_bar_area:.1f}mm²"
                    + f", total area: {self.secondary_area_total:.1f}mm² {' (unit)' if self.secondary_is_unit else ''}."
                )
                if self.is_mesh
                else ""
            )
        )


def reo_prop(
    bar_spec: str,
    *,
    main_width: float | None = None,
    secondary_width: float | None = None,
) -> ReoProperties:
    """
    Returns reinforcement properties from a standard bar specification.

    Parameters
    ----------
    bar_spec: str
        A standard Australian bar specification code. Can be a single bar (e.g. 'N12'),
        multiple bars (e.g. '2-N12'), bars with spacing (e.g. 'N12@200'), or mesh
        designation (e.g. 'SL82', 'F82').
    main_width: float | None, default = None
        The width in mm over which to calculate the area of the main bars.
        Used for calculating bar spacing and total area of reinforcement.
        If None is provided, main bar area is a unit area for mesh and bars that
        only have a spacing.
    secondary_width: float | None, default = None
        The width in mm over which to calculate the area of the secondary bars.
        Only used for mesh reinforcement to calculate cross bar spacing and area.
        If None is provided, the secondary bar area is a unit area.
    """

    # TODO: move as much of this into the ReoProperties class as possible,
    #  as a classmethod.

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
        bar_type = BarSpec(is_no_bars[4][:1])
        main_dia = int(is_no_bars[4][1:])

        no_main = 1 if is_no_bars[1] is None else is_no_bars[1][:-1]
        no_main = 1 if no_bars is None else int(no_main)

        ret_val = ReoProperties(
            bar_spec=bar_spec,
            is_bar=True,
            is_mesh=False,
            main_dia=main_dia,
            bar_type=bar_type,
            main_spacing=None,
            secondary_dia=None,
            secondary_spacing=None,
            no_main=no_main,
            no_secondary=None,
            main_width=main_width,
            secondary_width=None,
        )

    if is_bars_spacing:
        bar_type = BarSpec(is_bars_spacing[1][:1])
        main_dia = int(is_bars_spacing[1][1:])
        bar_spacing = float(is_bars_spacing[4])

        # TODO: this logic can be moved into the ReoProperties object if a private
        #  _no_main property is implemented to allow no-size specs, along with
        #  a calculated no_main property.
        no_main = main_width / bar_spacing if main_width is not None else None

        ret_val = ReoProperties(
            bar_spec=bar_spec,
            is_bar=True,
            is_mesh=False,
            main_dia=main_dia,
            bar_type=bar_type,
            main_spacing=bar_spacing,
            secondary_dia=None,
            secondary_spacing=None,
            no_main=no_main,
            no_secondary=None,
            main_width=main_width,
            secondary_width=None,
        )

    if mesh:
        mesh_data = MESH_DATA[bar_spec]

        main_dia = mesh_data["bar_dia"]
        pitch = mesh_data["pitch"]
        no_main = main_width / pitch if main_width is not None else None

        secondary_dia = mesh_data["cross_bar_dia"]
        cross_pitch = mesh_data["cross_pitch"]
        no_secondary = (
            secondary_width / cross_pitch if secondary_width is not None else None
        )

        barspec = BarSpec(mesh[2] if len(mesh[2]) == 1 else mesh[2][1:])

        ret_val = ReoProperties(
            bar_spec=bar_spec,
            is_bar=False,
            is_mesh=True,
            main_dia=main_dia,
            bar_type=barspec,
            main_spacing=pitch,
            secondary_dia=secondary_dia,
            secondary_spacing=cross_pitch,
            no_main=no_main,
            no_secondary=no_secondary,
            main_width=main_width,
            secondary_width=secondary_width,
        )

    return ret_val


def mesh_mass(
    *,
    main_diameter: float,
    main_spacing: float,
    secondary_diameter: float | None = None,
    secondary_spacing: float | None = None,
    steel_density: float = 7850,
    main_width: float | None = None,
    secondary_width: float | None = None,
):
    """
    Calculate the mass of steel mesh.

    Parameters
    ----------
    main_diameter : float
        Diameter of the main bars (in meters).
    main_spacing : float
        Spacing of the main bars (in meters).
    secondary_diameter : float, optional
        Diameter of the secondary bars (in meters).
        By default None (will use main_diameter).
    secondary_spacing : float, optional
        Spacing of the secondary bars (in meters).
        By default None (will use main_spacing).
    steel_density : float
        Density of the steel (in kg/m^3).
        By default 7850.
    main_width : float, optional
        Width of the main mesh (in meters).
        By default 1.000.
    secondary_width : float, optional
        Width of the secondary mesh (in meters).
        By default 1.000.

    Returns
    -------
    float
        The mass of the steel mesh (in kg).

    Examples
    --------
    >>> mesh_mass(0.0095, 0.200)
    5.6

    >>> mesh_mass(0.01, 0.1, secondary_diameter=0.005, secondary_spacing=0.2)
    92.508
    """

    if secondary_diameter is None:
        secondary_diameter = main_diameter

    if secondary_spacing is None:
        secondary_spacing = main_spacing

    if main_width is None:
        main_width = 1.0

    if secondary_width is None:
        secondary_width = 1.0

    main_volume = circle_area(diameter=main_diameter) * main_width / main_spacing
    secondary_volume = (
        circle_area(diameter=secondary_diameter) * secondary_width / secondary_spacing
    )

    return steel_density * (main_volume + secondary_volume)


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
