"""
File to contain functions for design steel members.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

import numpy as np
import polars as pl
from sectionproperties.pre import Geometry
from shapely import Polygon

from utilityscripts.reports.report import Variable
from utilityscripts.steel import as4100
from utilityscripts.steel.steel import (
    SteelGrade,
    i_section_df,
    steel_grades,
)


@dataclass
class UnitSystem:
    length: str = "m"
    length_scale: float = 1.0
    area: str = "m²"
    area_scale: float = 1.0
    volume: str = "m³"
    volume_scale: float = 1.0
    force: str = "N"
    force_scale: float = 1.0
    stress: str = "Pa"
    stress_scale: float = 1.0
    moment: str = "Nm"
    moment_scale: float = 1.0
    preferred_fmt: str = ".4j"


UnitSystem_SI = UnitSystem()

UnitSystem_AS_Steel = UnitSystem(
    length="mm",
    length_scale=1000,
    area="mm²",
    area_scale=1e6,
    volume="mm³",
    volume_scale=1e9,
    force="kN",
    force_scale=1e-3,
    moment="kNm",
    moment_scale=1e-3,
    stress="MPa",
    stress_scale=1e-6,
)


class HoleLocation(StrEnum):
    TOPLEFT = "top-left"
    TOPRIGHT = "top-right"
    BOTTOMLEFT = "bottom-left"
    BOTTOMRIGHT = "bottom-right"
    WEB = "web"


class SteelSection(ABC):
    """
    Store a cross-section that provides section properties compatible with steel codes.

    This is an abstract class,
    and requires implementation for each section type, e.g. I, C, SHS etc.
    """

    def __init__(self, *, section_name, steel: SteelGrade | None = None):
        """

        Parameters
        ----------
        section_name : str
            A name to give the section.
        steel : SteelGrade | None
            The steel material the section is made of. Can be None, but if so only
            the geometric properties of the section will be available and no design
            can be done.
        """

        self._section_name = section_name
        self._steel = steel
        self._geometry = None

    def _copy_with_new(self, **new_attributes) -> Self:
        """
        Function to copy the SteelSection instance but update specific attributes.

        The returned copy is a deepcopy.
        Note that the function replaces _geometry of the new instance with None.
        The user of the instance is responsible for recreating the geometry.

        Parameters
        ----------
        new_attributes : dict
            Any attributes to update as key:value pairs.

        Returns
        -------
        SteelSection
            A new instance of an SteelSection with updated attributes.
        """

        new_section = copy.deepcopy(self)

        for attr, value in new_attributes.items():
            setattr(new_section, attr, value)

        new_section._geometry = None

        return new_section

    @property
    def section_name(self):
        """
        The section name.

        Returns
        -------
        str
        """
        return self._section_name

    @property
    def steel(self) -> SteelGrade | None:
        """
        The steel material the section is made of.

        Returns
        -------
        SteelGrade | None
            The steel material the section is made of.
            Will be None if not provided.
        """

        return self._steel

    def add_steel(self, steel: SteelGrade | None) -> SteelSection:
        """
        Add a steel property to the section.

        Parameters
        ----------
        steel : SteelGrade | None
            The steel material the section is made of. Can be None, but if so only
            the geometric properties of the section will be available and no design
            can be done.

        Returns
        -------
        SteelSection
            A new section containing the new steel properties.
        """

        return self._copy_with_new(**{"_steel": steel})

    @property
    @abstractmethod
    def f_y_min(self) -> float:
        """
        The minimum yield strength of the section.

        Returns
        _______
        float
        """

        raise NotImplementedError()

    @property
    @abstractmethod
    def f_u_min(self) -> float:
        """
        The minimum ultimate strength of the section.

        Returns
        -------
        float
        """

        raise NotImplementedError()

    @abstractmethod
    def _make_geometry(self):
        """
        A method to make the geometry object.
        Should set the self._geometry attribute.
        """

        raise NotImplementedError

    @property
    def geometry(self) -> Geometry:
        """
        A sectionproperties Geometry object describing the shape.
        For complex sections can be a CompoundGeometry object
        as this is a sub-class of Geometry.

        Notes
        -----
        - This property should create the geometry if it has not already been created.
        - The returned section should not include any localised holes (e.g. bolt holes).
        - Geometry should be centred on its COG.

        Returns
        -------
        Geometry
        """

        if self._geometry is None:
            self._make_geometry()

        return self.geometry

    @property
    def geometry_net(self) -> Geometry:
        """
        A sectionproperties Geometry object describing the shape
        along with any holes. This is optional for
        derived classes to implement.

        Returns
        -------
        Geometry
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def area_gross(self) -> float:
        """
        The gross area of the section.

        Returns
        -------
        float
        """
        pass

    @property
    @abstractmethod
    def area_net(self) -> float:
        """
        The net area of the section.

        Returns
        -------
        float
        """
        pass

    def plot_geometry(self, *, net: bool = False):
        """
        Plot the geometry of the section.

        Parameters
        ----------
        net : bool, optional
            Plot the full section or the net section?
        """

        if net:
            self.geometry_net.plot_geometry()
        else:
            self.geometry.plot_geometry()


class ISection(SteelSection):
    def __init__(
        self,
        *,
        section_name,
        steel: SteelGrade | None,
        b_f: float | tuple[float, float],
        d: float,
        t_f: float | tuple[float, float],
        t_w: float,
        corner_detail: CornerDetail | None = None,
        corner_size: float = 0.0,
        n_r: int = 8,
    ):
        """
        Initialize an ISection object.

        Parameters
        ----------
        section_name : str
            The name of the section.
        steel : SteelGrade | None
            The steel material the section is made of. Can be None, but if so only
            the geometric properties of the section will be available and no design
            can be done.
        b_f : float | tuple[float, float]
            Width of the flange. For monosymmetric I sections,
            a tuple of the top & bottom flanges can be provided as (b_ft, b_fb)
        d : float
            Total depth of the section.
        t_f : float | tuple[float, float]
            Thickness of the flange. For monosymmetric I sections
            a tuple of the top & bottom flanges can be provided as (t_ft, t_fb)
        t_w : float
            Thickness of the web.
        corner_detail: CornerDetail | None
            What is the web-flange interface detail?
            A CornerDetail Enum or None.
            If None it is assumed there is a sharp 90deg angle.
        corner_size: float:
            What size is the corner detail? Ignored if corner_detail is None.
        n_r: int:
            The number of points used to make a corner radius.
            Only used if a corner radius is specified.
        """

        super().__init__(section_name=section_name, steel=steel)

        if isinstance(b_f, float):
            b_f = (b_f, b_f)

        if isinstance(t_f, float):
            t_f = (t_f, t_f)

        self._b_f = b_f
        self._d = d
        self._t_f = t_f
        self._t_w = t_w

        self._corner_detail = corner_detail

        if corner_detail is None:
            corner_size = None

        if corner_size is None:
            corner_size = 0.0

        self._corner_size = corner_size
        self._n_r = n_r

        self._holes = []
        self._geometry_net = None

    @property
    def f_yw(self) -> float:
        """
        The yield strength of the web.

        Returns
        -------
        float
        """

        return self.steel.get_f_y(thickness=self.t_w)

    @property
    def f_yft(self) -> float:
        """
        The yield strength of the top flange.

        Returns
        -------
        float
        """

        return self.steel.get_f_y(thickness=self.t_ft)

    @property
    def f_yfb(self) -> float:
        """
        The yield strength of the bottom flange.

        Returns
        -------
        float
        """

        return self.steel.get_f_y(thickness=self.t_fb)

    @property
    def f_uw(self) -> float:
        """
        The ultimate strength of the web.

        Returns
        -------
        float
        """

        return self.steel.get_f_u(thickness=self.t_w)

    @property
    def f_uft(self) -> float:
        """
        The ultimate strength of the top flange.

        Returns
        -------
        float
        """

        return self.steel.get_f_u(thickness=self.t_ft)

    @property
    def f_ufb(self) -> float:
        """
        The ultimate strength of the bottom flange.

        Returns
        -------
        float
        """

        return self.steel.get_f_u(thickness=self.t_fb)

    @property
    def f_y_min(self) -> float:
        """
        The minimum yield strength of the section.

        Returns
        _______
        float
        """

        return min(self.f_yw, self.f_yft, self.f_yfb)

    @property
    def f_u_min(self) -> float:
        """
        The minimum ultimate strength of the section.

        Returns
        -------
        float
        """

        return min(self.f_uw, self.f_uft, self.f_ufb)

    @property
    def b_f(self) -> tuple[float, float]:
        """
        Returns the flange widths as a tuple of (top, bottom)

        Returns
        -------
        tuple of float
            A tuple consisting of (b_ft, b_fb)
        """
        return self._b_f

    @property
    def b_ft(self) -> float:
        """
        The top flange width.

        Returns
        -------
        float
        """
        return self._b_f[0]

    @property
    def b_fb(self) -> float:
        """
        The bottom flange width.

        Returns
        -------
        float
        """
        return self._b_f[1]

    @property
    def d(self):
        return self._d

    @property
    def t_f(self) -> tuple[float, float]:
        """
        Returns the flange thicknesses as a tuple of (top, bottom)

        Returns
        -------
        tuple of float
            A tuple consisting of (t_ft, t_fb)
        """
        return self._t_f

    @property
    def t_ft(self) -> float:
        """
        The top flange thickness.

        Returns
        -------
        float
        """
        return self._t_f[0]

    @property
    def t_fb(self) -> float:
        """
        The bottom flange thickness.

        Returns
        -------
        float
        """
        return self._t_f[1]

    @property
    def t_w(self) -> float:
        """
        The web thickness.

        Returns
        -------
        float
        """
        return self._t_w

    @property
    def d_w(self) -> float:
        """
        Returns the clear web depth of the section.

        Returns
        -------
        float
        """
        return self.d - self.t_ft - self.t_fb

    @property
    def corner_detail(self):
        """
        The corner detail used for the I-section.
        If None, this assumes a sharp 90deg junction.
        Otherwise use a CornerDetail Enum to specify a weld
        or a radius.

        Returns
        -------
        CornerDetail | None
        """
        return self._corner_detail

    @property
    def corner_size(self):
        """
        The size of the corner detail (if any).

        Returns
        -------
        float | None
        """

        return self._corner_size

    @property
    def n_r(self):
        """
        The number of points used to make a corner radius.
        Only used if a radius is specified.

        Returns
        -------
        int
        """
        return self._n_r

    def _make_geometry(self):
        """
        A private method to make the geometry for the section.
        Sets the _geometry property.

        Notes
        -----
        This method does not include any localised holes (e.g. bolt holes).
        """

        poly = self._base_polygon()
        geom = Geometry(geom=poly).align_center()
        self._geometry = geom

    def _bottom_flange_poly(self) -> Polygon:
        points = [
            (self.b_fb / 2, 0),
            (self.b_fb / 2, self.t_fb),
            (-self.b_fb / 2, self.t_fb),
            (-self.b_fb / 2, 0),
        ]
        return Polygon(points)

    def _top_flange_poly(self) -> Polygon:
        points = [
            (self.b_ft / 2, self.d - self.t_ft),
            (self.b_ft / 2, self.d),
            (-self.b_ft / 2, self.d),
            (-self.b_ft / 2, self.d - self.t_ft),
        ]
        return Polygon(points)

    def _web_poly(self) -> Polygon:
        points = [
            (self.t_w / 2, self.t_fb),
            (self.t_w / 2, self.d - self.t_ft),
            (-self.t_w / 2, self.d - self.t_ft),
            (-self.t_w / 2, self.t_fb),
        ]

        return Polygon(points)

    def _base_polygon(self) -> Polygon:
        """
        A private method to make the base geometry for the section.

        Returns
        -------
        Polygon :
            A Polygon object representing the I Section.
        """

        points_a = [(self.b_fb / 2, 0), (self.b_fb / 2, self.t_fb)]

        cnr_btm_right = _make_corner(
            corner_point=(self.t_w / 2, self.t_fb),
            corner_size=self.corner_size,
            corner_location=CornerLocation.BOTTOMRIGHT,
            corner_detail=self.corner_detail,
            n_r=self.n_r,
        )
        cnr_top_right = _make_corner(
            corner_point=(self.t_w / 2, self.d - self.t_fb),
            corner_size=self.corner_size,
            corner_location=CornerLocation.TOPRIGHT,
            corner_detail=self.corner_detail,
            n_r=self.n_r,
        )
        points_b = cnr_btm_right + cnr_top_right

        points_c = [
            (self.b_ft / 2, self.d - self.t_ft),
            (self.b_ft / 2, self.d),
            (-self.b_ft / 2, self.d),
            (-self.b_ft / 2, self.d - self.t_ft),
        ]

        cnr_top_left = _make_corner(
            corner_point=(-self.t_w / 2, self.d - self.t_ft),
            corner_size=self.corner_size,
            corner_location=CornerLocation.TOPLEFT,
            corner_detail=self.corner_detail,
            n_r=self.n_r,
        )
        cnr_bottom_left = _make_corner(
            corner_point=(-self.t_w / 2, self.t_fb),
            corner_size=self.corner_size,
            corner_location=CornerLocation.BOTTOMLEFT,
            corner_detail=self.corner_detail,
            n_r=self.n_r,
        )
        points_d = cnr_top_left + cnr_bottom_left

        points_e = [(-self.b_fb / 2, self.t_fb), (-self.b_fb / 2, 0)]

        all_points = points_a + points_b + points_c + points_d + points_e
        return Polygon(all_points)

    @property
    def geometry(self) -> Geometry:
        """
        A geometry object representing the I Section.

        Returns
        -------
        Geometry
        """

        if self._geometry is None:
            self._make_geometry()

        return self._geometry

    @property
    def holes(self) -> list[tuple[HoleLocation, tuple[float, float]]]:
        """
        The holes in the section.

        Returns :
        -------
        list[tuple[HoleLocation, tuple[float, float]]]
            A list of holes, where each hole is a tuple of:
            (hole_location, (diameter, location))
        """

        return self._holes

    def add_holes(
        self, holes: list[tuple[HoleLocation, tuple[float, float]]]
    ) -> ISection:
        """
        Add holes to the section.

        Parameters
        ----------
        holes : list[tuple[HoleLocation, tuple[float, float]]]
            A list of holes, where each hole is a tuple of:
            (hole_location, (diameter, offset from CL))
            Valid locations are:
                "top-left", "top-right", "bottom-left", "bottom-right", "web"
            The offset from the CL should be +ve for the flanges.
            For the web it can be either +ve or -ve.

        Returns
        -------
        ISection
            A new ISection object with the added holes.
        """

        for _loc, dims in holes:
            if dims[1] < 0.0:
                raise ValueError(
                    "Expected Centreline of hole to be positive for all holes. "
                    + f"Hole located at {dims[1]} has offset {dims[1]} < 0.0"
                )

        return self._copy_with_new(**{"_holes": self.holes + holes})

    @property
    def area_gross(self) -> float:
        return self.geometry.calculate_area()

    @property
    def area_net(self) -> float:
        raise NotImplementedError()


class SteelMember:
    def __init__(
        self,
        *,
        section: SteelSection | None = None,
        length: float | None = None,
        restraints=None,
    ):
        self._section = section
        self._length = length
        self._restraints = restraints

    def _copy_with_new(self, **new_attributes) -> SteelMember:
        """
        Function to copy the SteelMember instance but update specific attributes.

        The returned copy is a deepcopy.

        Parameters
        ----------
        new_attributes : dict
            Any attributes to update as key:value pairs.

        Returns
        -------
        SteelMember
            A new instance of a SteelMember with updated attributes.
        """

        new_member = copy.deepcopy(self)

        for attr, value in new_attributes.items():
            setattr(new_member, attr, value)

        return new_member

    @property
    def section(self) -> SteelSection | None:
        """
        The cross-section of the member.

        Returns
        -------
        SteelSection | None
            The current assigned cross-section object.

        Notes
        -------
        In future it is expected that this may be extended to handle
        multiple different cross-sections along the length of a member.
        """
        return self._section

    def add_section(self, section: SteelSection | None) -> SteelMember:
        """
        Add or replace the section in the member.

        Parameters
        ----------
        section : SteelSection | None
            The new section to add.

        Returns
        -------
        SteelMember
            A new SteelMember with updated attributes.
        """

        return self._copy_with_new(**{"_section": section})

    @property
    def length(self):
        return self._length

    def add_length(self, length: float | None) -> SteelMember:
        """
        Add or replace the length of the member.

        Parameters
        ----------
        length : float | None
            The new length to add.

        Returns
        -------
        SteelMember
            A new SteelMember with updated attributes.
        """

        return self._copy_with_new(**{"_length": length})

    @property
    def restraints(self):
        return self._restraints


class AS4100:
    """
    A class for design to AS4100.
    """

    def __init__(
        self,
        *,
        member: SteelMember | None = None,
        unit_system: UnitSystem = UnitSystem_AS_Steel,
    ):
        """
        A class for design to AS4100.

        Parameters
        ----------
        member : SteelMember
            The steel member object to design.
        unit_system : UnitSystem
            A UnitSystem object to use for formatting output results.
        """

        self._member = member
        self._unit_system = unit_system

    def _copy_with_new(self, **new_attributes) -> AS4100:
        """
        Function to copy the instance but update specific attributes.

        The returned copy is a deepcopy.

        Parameters
        ----------
        new_attributes : dict
            Any attributes to update as key:value pairs.

        Returns
        -------
        AS4100
            A new instance of AS4100 with updated attributes.
        """

        new_design = copy.deepcopy(self)

        for attr, value in new_attributes.items():
            setattr(new_design, attr, value)

        return new_design

    @property
    def member(self) -> SteelMember | None:
        """
        The member assigned to the object.

        Returns
        -------
        SteelMember | None
            The steel member assigned to the object.
        """

        return self._member

    def add_member(self, member: SteelMember) -> AS4100:
        """
        Add or replace the member in the member.

        Parameters
        ----------
        member : SteelMember
            The member to be added.

        Returns
        -------
        AS4100
            A new AS4100 with updated attributes.
        """

        return self._copy_with_new(**{"_member": member})

    @property
    def unit_system(self):
        return self._unit_system

    @property
    def length(self) -> float | None:
        """
        The length of the member assigned to the design.

        Returns
        -------
        float | None
            The length of the member assigned to the design.
        """

        return None if self.member is None else self.member.length

    @property
    def section(self) -> SteelSection:
        return self.member.section

    def n_ty(self) -> Variable:
        """
        The tension yield capacity. In kN.

        Returns
        -------
        float
        """

        return Variable(
            as4100.s7_2_n_ty(a_g=self.section.area_gross, f_y=self.section.f_y_min),
            symbol=("N_ty", "N_{ty}"),
            scale=self.unit_system.force_scale,
            units=self.unit_system.force,
            fmt_string=self.unit_system.preferred_fmt,
        )

    def phi_n_ty(self, phi_steel: float = 0.9) -> Variable:
        """
        Calculate the design tension yield capacity (φN_ty).

        Parameters
        ----------
        phi_steel : float, optional
            The capacity reduction factor for steel in tension, default is 0.9.

        Returns
        -------
        float
        """

        return Variable(
            self.n_ty().value * phi_steel,
            symbol=("φN_ty", "\\phi N_{ty}"),
            scale=self.unit_system.force_scale,
            units=self.unit_system.force,
            fmt_string=self.unit_system.preferred_fmt,
        )

    def n_tu(self, fracture_modifier: float = 0.85) -> Variable:
        """
        Calculate the net tensile strength at fracture.

        Parameters
        ----------
        fracture_modifier : float, optional
            A modifier representing the impact of fractures or imperfections.
            Default value is 0.85.

        Returns
        -------
        float
        """
        return Variable(
            self.section.area_net * self.section.f_u_min * fracture_modifier,
            scale=0.001,
            units="kN",
            fmt_string=self.unit_system.preferred_fmt,
        )

    def phi_n_tu(
        self, fracture_modifier: float = 0.85, phi_steel: float = 0.9
    ) -> Variable:
        """
        Calculate the design fracture capacity.

        Parameters
        ----------
        fracture_modifier : float
            Modifier applied to the ultimate capacity factor (default is 0.85).
        phi_steel : float
            Capacity reduction factor for steel (default is 0.9).

        Returns
        -------
        float
        """

        return Variable(
            self.n_tu(fracture_modifier=fracture_modifier).value * phi_steel,
            scale=0.001,
            units="kN",
            fmt_string=self.unit_system.preferred_fmt,
        )


class CornerDetail(StrEnum):
    """
    What sort of web-flange interface does the section have?
    """

    WELD = "weld"
    RADIUS = "radius"


class CornerLocation(StrEnum):
    """
    Provides location information for where a corner is located on a section.
    """

    TOPLEFT = "topleft"
    TOPRIGHT = "topright"
    BOTTOMLEFT = "bottomleft"
    BOTTOMRIGHT = "bottomright"


def _make_corner(
    *,
    corner_point: tuple[float, float],
    corner_size: float,
    corner_location: CornerLocation,
    corner_detail: CornerDetail | None = None,
    n_r: int = 8,
):
    """
    Generate points for the corners of the section based on the corner detail & size.

    Parameters
    ----------
    corner_point : tuple of float
        The original corner point coordinates as a tuple (x, y).
    corner_size : float
        The radius size or weld leg length.
    right : bool
        Direction flag indicating whether the corner is at the right.
    top : bool
        Direction flag indicating whether the corner is at the top
    corner_detail : CornerDetail | None
        The detail used in the corner.
        If None a sharp 90deg corner is assumed.
    n_r : int
        The number of points around a radius.

    Returns
    -------
    list[tuple[float, float]]
        A list containing the new corner point coordinates.
    """
    if corner_detail is None:
        return [corner_point]

    if corner_size is None or corner_size == 0:
        return [corner_point]

    if corner_detail == CornerDetail.WELD:
        return _make_weld(
            corner_point=corner_point,
            corner_size=corner_size,
            corner_location=corner_location,
        )

    return _make_radius(
        corner_point=corner_point,
        corner_size=corner_size,
        corner_location=corner_location,
        n_r=n_r,
    )


def _make_weld(
    *,
    corner_point: tuple[float, float],
    corner_size: float,
    corner_location: CornerLocation,
):
    """
    Create coordinates for a corner weld.

    Parameters
    ----------
    corner_point : tuple
        The original corner point (x, y).
    corner_size : float
        The leg length of the weld.
    corner_location : CornerLocation
        The corner location (e.g. Top Left, Bottom Right).

    Returns
    -------
    list[tuple[float, float]]
        A list containing the new corner point coordinates.
    """

    if corner_location == CornerLocation.TOPRIGHT:
        return [
            (corner_point[0], corner_point[1] - corner_size),
            (corner_point[0] + corner_size, corner_point[1]),
        ]
    if corner_location == CornerLocation.BOTTOMRIGHT:
        return [
            (corner_point[0] + corner_size, corner_point[1]),
            (corner_point[0], corner_point[1] + corner_size),
        ]
    if corner_location == CornerLocation.TOPLEFT:
        return [
            (corner_point[0] - corner_size, corner_point[1]),
            (corner_point[0], corner_point[1] - corner_size),
        ]

    return [
        (corner_point[0], corner_point[1] + corner_size),
        (corner_point[0] - corner_size, corner_point[1]),
    ]


def _make_radius(
    *,
    corner_point: tuple[float, float],
    corner_size: float,
    corner_location: CornerLocation,
    n_r: int = 8,
) -> list[tuple[float, float]]:
    """
    Generate a list of points representing a circle segment radius for a given
    corner based on the specified position (right, top).

    Parameters
    ----------
    corner_point :
        The original corner point.
    corner_size : float
        The size of the radius
    corner_location : CornerLocation
        The corner location (e.g. Top Left, Bottom Right).
    n_r : int
        The number of points around the radius

    Returns
    -------
    list[tuple[float, float]]
        A list containing the new corner point coordinates.
    """

    match corner_location:
        case CornerLocation.TOPRIGHT:
            radius_centre = (
                corner_point[0] + corner_size,
                corner_point[1] - corner_size,
            )
            limit_angles = (90, 180)
        case CornerLocation.BOTTOMRIGHT:
            radius_centre = (
                corner_point[0] + corner_size,
                corner_point[1] + corner_size,
            )
            limit_angles = (180, 270)
        case CornerLocation.TOPLEFT:
            radius_centre = (
                corner_point[0] - corner_size,
                corner_point[1] - corner_size,
            )
            limit_angles = (0, 90)
        case CornerLocation.BOTTOMLEFT:
            radius_centre = (
                corner_point[0] - corner_size,
                corner_point[1] + corner_size,
            )
            limit_angles = (270, 360)
        case _:
            raise ValueError(f"Invalid corner location: {corner_location}")

    radius = build_circle(
        centroid=radius_centre,
        radius=corner_size,
        no_points=n_r,
        limit_angles=limit_angles,
        use_radians=False,
    )

    points = list(reversed(radius))  # need to reverse points from clockwise to anti.

    return [corner_point, *points]


def build_circle(
    *,
    centroid: tuple[float, float],
    radius: float,
    no_points: int = 64,
    limit_angles: tuple[float, float] | None = None,
    use_radians: bool = True,
) -> list[tuple[float, float]]:
    """
    Build a list of points that approximate a circle or circular arc.
    Typically used to create web-flange radii.

    centroid: tuple[float, float]
        The centroid of the circle.
    radius: float
        The radius of the circle.
    no_points: int
        The no. of points to include in the definition of the circle.
    limit_angles: tuple[float, float] | None
     Angles to limit the circular arc. Should be of the format
        (min, max).
        Angles to be taken CCW.
    use_radians: bool
        Use radians for angles?

    Returns
    -------
    list[tuple[float, float]]
        A circle, or part thereof, as a list of lists defining the points:
        [[x1, y1], [x2, y2], ..., [xn, yn]]
    """

    full_circle = math.radians(360)

    if limit_angles is not None:
        min_angle = limit_angles[0]
        max_angle = limit_angles[1]

        if not use_radians:
            min_angle = math.radians(min_angle)
            max_angle = math.radians(max_angle)

    else:
        min_angle = 0
        max_angle = full_circle

    if min_angle == 0 and max_angle == full_circle:
        no_points += 1  # when the start and end points overlap, add a point.

    angle_range = np.linspace(start=min_angle, stop=max_angle, num=no_points)
    x_points_orig = np.full(no_points, radius)

    x_points = x_points_orig * np.cos(angle_range)  # - y_points * np.sin(angle_range)
    y_points = x_points_orig * np.sin(angle_range)  # + y_points * np.cos(angle_range)
    # can neglect the 2nd half of the formula because the y-points are just zeroes

    x_points = x_points + centroid[0]
    y_points = y_points + centroid[1]

    all_points = np.transpose(np.stack((x_points, y_points)))

    return [(p[0], p[1]) for p in all_points.tolist()]


def make_section(
    *, designation: str, grade: str | SteelGrade | None = None
) -> SteelSection:
    """
    Make a SteelSection object based on the standard designation.

    Parameters
    ----------
    designation: str
        The standard designation of the section. E.g. "310UB40.4". M
        Must match the sections in the standard database.
    grade: str | SteelGrade | None
        The grade of steel to be used. Can be either a standard specification string
        matching those in the data spreadsheet (e.g. 'AS/NZS3678:250') or a SteelGrade
        object. If None, the SteelSection object has no material properties.

    Returns
    -------
    SteelSection
        A SteelSection object representing the specified section.
    """

    i_df = i_section_df()

    if designation not in i_df["section"]:
        raise ValueError(f"Invalid section designation: {designation}")

    row = i_df.filter(pl.col("section") == designation)

    if grade is not None and not isinstance(grade, SteelGrade):
        grade = steel_grades()[grade]

    corner_detail = (
        CornerDetail.WELD
        if row["fabrication_type"][0] == "welded"
        else CornerDetail.RADIUS
    )
    corner_size = row["r1_or_w1"][0]

    return ISection(
        section_name=row["section"][0],
        steel=grade,
        b_f=row["b_f"][0],
        d=row["d"][0],
        t_f=row["t_f"][0],
        t_w=row["t_w"][0],
        corner_detail=corner_detail,
        corner_size=corner_size,
    )
