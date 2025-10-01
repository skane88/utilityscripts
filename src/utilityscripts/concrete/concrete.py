"""
File to contain some utilities for working with concrete.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from math import pi, sin

import matplotlib.pyplot as plt
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
from shapely import LineString, Polygon, ops
from shapely.plotting import plot_polygon

from utilityscripts.concrete.as3600 import Concrete, Steel
from utilityscripts.steel.design import build_circle

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


class Cover(StrEnum):
    """
    Enum to represent different types of cover.
    """

    TOP = "top"
    BOTTOM = "bottom"
    SIDE = "side"
    ALL = "all"


class RectBeam:
    """
    Class to represent a simple rectangular beam.

    Notes
    -----
    - At this point only top, bottom & side reinforcement will be considered.
    - Only rectangular sections will be considered.
    - All steel assumed to be the same grade.
    - All reo is assumed to be evenly distributed across the relevant face.

    For more complex sections recommend using the `concreteproperties` package.
    """

    def __init__(
        self,
        *,
        b: float,
        d: float,
        concrete: Concrete | float,
        steel: Steel | float,
        cover: float | dict[Cover, float],
        bot_dia: float | None = None,
        bot_no: float | None = None,
        top_dia: float | None = None,
        top_no: float | None = None,
        side_dia: float | None = None,
        side_no: float | None = None,
        shear_dia: float | None = None,
        shear_no_ligs: float | None = None,
        shear_spacing: float | None = None,
        shear_steel: Steel | float | None = None,
    ):
        """
        Initialize the RectBeam class.

        Parameters
        ----------
        b : float
            The width of the beam.
        d : float
            The depth of the beam.
        concrete : Concrete | float
            A Concrete material object or the characteristic compressive strength.
        steel : Steel | float
            A Steel material object or the yield strength.
        cover : float | dict[str, float]
            The cover to the reinforcement.
            If a float is provided, it is assumed to be the same for all reinforcement.
            If a dictionary is provided, it should contain the keys
            "bot", "top" & "side".
        bot_dia : float
            The diameter of the bottom reinforcement.
        bot_no : float
            The number of bottom reinforcement bars.
        top_dia : float
            The diameter of the top reinforcement.
        top_no : float
            The number of top reinforcement bars.
        side_dia : float
            The diameter of the side reinforcement.
        side_no : float
            The number of side reinforcement bars.
        shear_dia : float
            The diameter of the shear reinforcement.
        shear_no_ligs : float
            The number of legs of shear reinforcement.
        shear_spacing : float
            The spacing of the shear reinforcement.
        shear_steel : Steel | None
            The shear reinforcement.
        """

        self._b = b
        self._d = d

        if isinstance(concrete, (int, float)):
            self._concrete = Concrete(f_c=concrete)
        else:
            self._concrete = concrete

        if isinstance(steel, (int, float)):
            self._steel = Steel(f_sy=steel)
        else:
            self._steel = steel

        self._bot_dia = bot_dia
        self._bot_no = bot_no
        self._top_dia = top_dia
        self._top_no = top_no
        self._side_dia = side_dia
        self._side_no = side_no
        self._shear_dia = shear_dia
        self._shear_no_ligs = shear_no_ligs
        self._shear_spacing = shear_spacing

        if isinstance(shear_steel, (int, float)):
            self._shear_steel = Steel(f_sy=shear_steel)
        else:
            self._shear_steel = shear_steel

        self._cover = {}

        if isinstance(cover, (int, float)):
            self._cover[Cover.TOP] = float(cover)
            self._cover[Cover.BOTTOM] = float(cover)
            self._cover[Cover.SIDE] = float(cover)
        else:
            self._cover[Cover.BOTTOM] = (
                cover[Cover.BOTTOM] if Cover.BOTTOM in cover else cover[Cover.ALL]
            )
            self._cover[Cover.TOP] = (
                cover[Cover.TOP] if Cover.TOP in cover else cover[Cover.ALL]
            )
            self._cover[Cover.SIDE] = (
                cover[Cover.SIDE] if Cover.SIDE in cover else cover[Cover.ALL]
            )

    @property
    def b(self):
        """
        Return the width of the beam.
        """

        return self._b

    @property
    def d(self):
        """
        Return the depth of the beam.
        """

        return self._d

    @property
    def concrete(self) -> Concrete:
        """
        Return a Concrete material object.
        """

        return self._concrete

    @property
    def steel(self) -> Steel:
        """
        Return a Steel material object.
        """

        return self._steel

    @property
    def modular_ratio(self):
        """
        Return the modular ratio of the steel & concrete.
        """

        return self.steel.elastic_modulus / self.concrete.elastic_modulus

    @property
    def top_dia(self):
        """
        Return the diameter of the top reinforcement.
        """

        if self._top_dia is None:
            return 0

        return self._top_dia

    @property
    def y_top(self):
        """
        Return the y-coordinate of the top reinforcement.
        """

        return self.d - self.top_cover - self.shear_dia - self.top_dia / 2

    @property
    def top_bar_area(self):
        """
        Return the area of a single top reinforcement bar.
        """

        return circle_area(self.top_dia)

    @property
    def top_no(self):
        """
        Return the number of top reinforcement bars.
        """

        return self._top_no

    @property
    def top_steel_area(self):
        """
        Return the total area of the top reinforcement.
        """

        if self.top_no is None:
            return 0.0

        return self.top_bar_area * self.top_no

    @property
    def top_cover(self):
        """
        Return the cover to the top reinforcement.
        """

        return self._cover[Cover.TOP]

    @property
    def top_bar_centres(self) -> list[tuple[float, float]]:
        """
        Return the centres of the top reinforcement bars.
        """

        if self.top_no is None or self.top_dia == 0:
            return []

        if self.top_no == 1:
            return [(0, self.y_top)]

        x_min = -self.b / 2 + (self.side_cover + self.shear_dia + self.top_dia / 2)
        x_max = self.b / 2 - (self.side_cover + self.shear_dia + self.top_dia / 2)
        centre_to_centre = (x_max - x_min) / (self.top_no - 1)

        return [(x_min + i * centre_to_centre, self.y_top) for i in range(self.top_no)]

    @property
    def bot_dia(self):
        """
        Return the diameter of the bottom reinforcement.
        """

        if self._bot_dia is None:
            return 0

        return self._bot_dia

    @property
    def y_bot(self):
        """
        Return the y-coordinate of the bottom reinforcement.
        """

        return self.bot_cover + self.shear_dia + self.bot_dia / 2

    @property
    def bot_bar_area(self):
        """
        Return the area of a single bottom reinforcement bar.
        """

        return circle_area(self.bot_dia)

    @property
    def bot_no(self):
        """
        Return the number of bottom reinforcement bars.
        """

        return self._bot_no

    @property
    def bot_steel_area(self):
        """
        Return the total area of the bottom reinforcement.
        """

        if self.bot_no is None:
            return 0.0

        return self.bot_bar_area * self.bot_no

    @property
    def bot_cover(self):
        """
        Return the cover to the bottom reinforcement.
        """

        return self._cover[Cover.BOTTOM]

    @property
    def bot_bar_centres(self) -> list[tuple[float, float]]:
        """
        Return the centres of the bottom reinforcement bars.
        """

        if self.bot_dia == 0 or self.bot_no is None:
            return []

        if self.bot_no == 1:
            return [(0, self.y_bot)]

        x_min = -self.b / 2 + (self.side_cover + self.shear_dia + self.bot_dia / 2)
        x_max = self.b / 2 - (self.side_cover + self.shear_dia + self.bot_dia / 2)
        centre_to_centre = (x_max - x_min) / (self.bot_no - 1)

        return [(x_min + i * centre_to_centre, self.y_bot) for i in range(self.bot_no)]

    @property
    def side_dia(self):
        """
        Return the diameter of the side reinforcement.
        """

        return self._side_dia

    @property
    def side_bar_area(self):
        """
        Return the area of a single side reinforcement bar.
        """

        return circle_area(self.side_dia)

    @property
    def side_no(self):
        """
        Return the number of side reinforcement bars per side.
        """

        return self._side_no

    @property
    def side_steel_area(self):
        """
        Return the total area of the side reinforcement.
        """

        if self.side_no is None:
            return 0.0

        return self.side_bar_area * self.side_no * 2

    @property
    def side_cover(self):
        """
        Return the cover to the side reinforcement.
        """

        return self._cover[Cover.SIDE]

    @property
    def side_bar_centres(self) -> list[tuple[float, float]]:
        """
        Return the centres of the side reinforcement bars.
        """

        if self.side_dia == 0 or self.side_no is None:
            return []

        y_bot = self.y_bot + self.bot_dia / 2
        y_top = self.y_top - self.top_dia / 2

        gap = y_top - y_bot
        centre_to_centre = gap / (self.side_no + 1)

        x_left = -self.b / 2 + (self.side_cover + self.shear_dia + self.side_dia / 2)
        x_right = self.b / 2 - (self.side_cover + self.shear_dia + self.side_dia / 2)

        return [
            (x_left, y_bot + (i + 1) * centre_to_centre) for i in range(self.side_no)
        ] + [(x_right, y_bot + (i + 1) * centre_to_centre) for i in range(self.side_no)]

    @property
    def area_steel(self):
        """
        Return the total area of the reinforcement.
        """

        return self.top_steel_area + self.bot_steel_area + self.side_steel_area

    @property
    def shear_dia(self):
        """
        Return the diameter of the shear reinforcement.
        """

        if self._shear_dia is None:
            return 0

        return self._shear_dia

    @property
    def shear_bar_area(self):
        """
        Return the area of the shear reinforcement bars.
        """

        return circle_area(self.shear_dia)

    @property
    def shear_no_ligs(self):
        """
        Return the number of legs of shear reinforcement.
        """

        if self._shear_no_ligs is None:
            return 0

        return self._shear_no_ligs

    @property
    def area_steel_shear(self):
        """
        Return the total area of the shear reinforcement.
        """

        return self.shear_bar_area * self.shear_no_ligs

    @property
    def shear_spacing(self):
        """
        Return the spacing of the shear reinforcement.
        """

        return self._shear_spacing

    @property
    def shear_steel(self) -> Steel:
        """
        Return the shear reinforcement steel object.
        """

        if self._shear_steel is None:
            self._shear_steel = self.steel

        return self._shear_steel

    @property
    def area_gross(self):
        """
        Return the gross area of the beam.
        """

        return self.b * self.d

    @property
    def transformed_area(self):
        """
        Return the transformed area of the beam.
        """

        return (
            self.area_gross - self.area_steel
        ) + self.area_steel * self.modular_ratio

    @property
    def _geometry_points(self):
        """
        Create a series of points representing the beam.

        This method is intended to be a helper method for calculating member properties.

        Returns
        -------
        dict[str, tuple[float, float]]

        Where the keys are "concrete" and "steel".
        """

        no_points = 8

        # The area of a polygon is slightly smaller than the area of a circle.
        # Therefore factor the area of the reinforcement polygon up to give an
        # equivalent area to the circular bar.
        alpha = 2 * pi / no_points
        beta = ((alpha) / (sin(alpha))) ** 0.5

        steel = [
            build_circle(
                centroid=bar, radius=beta * self.top_dia / 2, no_points=no_points
            )
            for bar in self.top_bar_centres
        ]
        steel += [
            build_circle(
                centroid=bar, radius=beta * self.bot_dia / 2, no_points=no_points
            )
            for bar in self.bot_bar_centres
        ]
        steel += [
            build_circle(
                centroid=bar, radius=beta * self.side_dia / 2, no_points=no_points
            )
            for bar in self.side_bar_centres
        ]

        concrete = [
            (-self.b / 2, 0),
            (self.b / 2, 0),
            (self.b / 2, self.d),
            (-self.b / 2, self.d),
        ]

        return {"steel": steel, "concrete": concrete}

    @property
    def _base_concrete_geometry(self):
        """
        Return the basic geometry of the concrete.
        """

        poly = Polygon(self._geometry_points["concrete"])

        for bar in self._geometry_points["steel"]:
            steel = Polygon(bar)
            poly = poly - steel

        return poly

    def concrete_geometry(self, y: float = 0.0) -> list[Polygon]:
        """
        Return a geometry object that represents the concrete. Concrete can be
        cut through by a line at a given y-coordinate.
        Elements above the cut will be returned.

        Parameters
        ----------
        y: float
            The y-coordinate to cut the concrete at.

        Returns
        -------
        list[Polygon]
            A list of polygons representing the concrete.
        """

        length = self.b * 1.1

        cut_line = LineString([(-length / 2, y), (length / 2, y)])

        split_sect = ops.split(self._base_concrete_geometry, cut_line)

        return [sect for sect in split_sect.geoms if sect.centroid.y > y]

    def plot_concrete(self):
        plot_polygon(self.concrete_geometry()[0])
        plt.show()
