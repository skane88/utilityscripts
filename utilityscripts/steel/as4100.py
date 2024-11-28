"""
File to contain functions based on Australian Standard AS4100.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from sectionproperties.pre.geometry import Geometry
from shapely import Polygon


class CornerDetail(Enum):
    WELD = "weld"
    RADIUS = "radius"


class AS4100Section(ABC):
    """
    Store a cross-section that provides AS4100 compatible section properties.

    This is an abstract class,
    and requires implementation for each section type, e.g. I, C, SHS etc.
    """

    def __init__(self, *, section_name, f_y: float, f_u: float):
        """

        Parameters
        ----------
        section_name : str
            A name to give the section.
        f_y : float
            The section's yield strength.
        f_u : float
            The section's ultimate strength.
        """

        self._section_name = section_name
        self._f_y = f_y
        self._f_u = f_u
        self._geometry = None

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
    def f_y(self):
        """
        The yield strength of the section.

        Returns
        _______
        float
        """
        return self._f_y

    @property
    def f_u(self):
        """
        The ultimate strength of the section.

        Returns
        -------
        float
        """
        return self._f_u

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

    def n_ty(self) -> float:
        """
        The tension yield capacity.

        Returns
        -------
        float
        """
        return self.area_gross * self.f_y

    def phi_n_ty(self, phi_steel: float = 0.9) -> float:
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
        return self.n_ty() * phi_steel

    def n_tu(self, fracture_modifier: float = 0.85):
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
        return self.area_net * self.f_u * fracture_modifier

    def phi_n_tu(self, fracture_modifier: float = 0.85, phi_steel: float = 0.9):
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
        return self.n_tu(fracture_modifier=fracture_modifier) * phi_steel


class ISection(AS4100Section):
    def __init__(
        self,
        *,
        section_name,
        f_y: float,
        f_u: float,
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
        b_f : float | tuple[float, float]
            Width of the flange. For monosymmetric I sections,
            a tuple of the top & bottom flanges can be provided.
        d : float
            Total depth of the section.
        t_f : float | tuple[float, float]
            Thickness of the flange. For monosymmetric I sections
            a tuple of the top & bottom flanges can be provided.
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

        super().__init__(section_name=section_name, f_y=f_y, f_u=f_u)

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
        elif corner_size is None:
            corner_size = 0.0

        self._corner_size = corner_size
        self._n_r = n_r

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
        """

        points_a = [(self.b_fb / 2, 0), (self.b_fb / 2, self.t_fb)]

        cnr_btm_right = _make_corner(
            corner_point=(self.t_w / 2, self.t_fb),
            corner_size=self.corner_size,
            right=True,
            top=False,
            corner_detail=self.corner_detail,
            n_r=self.n_r,
        )
        cnr_top_right = _make_corner(
            corner_point=(self.t_w / 2, self.d - self.t_fb),
            corner_size=self.corner_size,
            right=True,
            top=True,
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
            right=False,
            top=True,
            corner_detail=self.corner_detail,
            n_r=self.n_r,
        )
        cnr_bottom_left = _make_corner(
            corner_point=(-self.t_w / 2, self.t_fb),
            corner_size=self.corner_size,
            right=False,
            top=False,
            corner_detail=self.corner_detail,
            n_r=self.n_r,
        )
        points_d = cnr_top_left + cnr_bottom_left

        points_e = [(-self.b_fb / 2, self.t_fb), (-self.b_fb / 2, 0)]

        all_points = points_a + points_b + points_c + points_d + points_e
        poly = Polygon(all_points)
        self._geometry = Geometry(geom=poly)

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
    def area_gross(self) -> float:
        return self.geometry.calculate_area()

    @property
    def area_net(self) -> float:
        raise NotImplementedError()


class AS4100Member:
    def __init__(self, section: AS4100Section, length, restraints):
        self._section = section
        self._length = length
        self._restraints = restraints

    @property
    def section(self) -> AS4100Section:
        """
        The cross-section of the member.

        Returns
        -------
        _any
            The current assigned cross-section object.

        Notes
        -------
        In future it is expected that this may be extended to handle
        multiple different cross-sections along the length of a member.
        """
        return self._section

    @property
    def length(self):
        return self._length

    @property
    def restraints(self):
        return self._restraints


def alpha_m(*, m_m, m_2, m_3, m_4):
    """
    Determines the moment modification factor as per AS4100 S5.6.1.1.a.iii.

    Parameters
    ----------
    m_m : float
        The maximum moment.
    m_2 : float
        The moment at the 1st 1/4 point.
    m_3 : float
        The moment at mid-span.
    m_4 : float
        The moment at the 2nd 1/4 point.

    Returns
    -------
    float
        The moment modification factor.
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

    This is used to reduce the shear capacity of a web when it is prone to buckling.

    Parameters
    ----------
    d_p : float
        The depth of the web panel.
    t_w : float
        The thickness of the web.
    f_y : float
        The yield strength of the web panel in MPa.
        If different units are used, make sure to update `f_y_ref` accordingly.
    s : float
        The length of the web or spacing between vertical stiffeners
        that meet the requirements of AS4100.
    f_y_ref : float, optional
        The reference yield stress, nominally 250.0.
        Default is 250.0MPa.

    Returns
    -------
    float
        The stiffened web shear buckling parameter alpha_v.

    Notes
    --------
    Units should be consistent across the inputs.
    E.g. all length units should be the same, and f_y and f_y_ref should match.
    If you

    References
    --------
    ..
    [1] Standards Australia, AS4100-2020+A2 (2024) Steel Structures

    Examples
    --------
    >>> d_p = 1.000  # Depth of the web panel
    >>> t_w = 0.008    # Thickness of the web
    >>> f_y = 300.0   # Yield strength of the web panel in MPa
    >>> s = 0.800     # Length of the web or spacing of vertical stiffeners
    >>> alpha_v(d_p=d_p, t_w=t_w, f_y=f_y, s=s)
    0.918947
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

    angle_range = np.linspace(start=min_angle, stop=max_angle, num=no_points)
    x_points_orig = np.full(no_points, radius)

    x_points = x_points_orig * np.cos(angle_range)  # - y_points * np.sin(angle_range)
    y_points = x_points_orig * np.sin(angle_range)  # + y_points * np.cos(angle_range)
    # can neglect the 2nd half of the formula because the y-points are just zeroes

    x_points = x_points + centroid[0]
    y_points = y_points + centroid[1]

    all_points = np.transpose(np.stack((x_points, y_points)))

    return [(p[0], p[1]) for p in all_points.tolist()]


def _make_corner(
    *,
    corner_point: tuple[float, float],
    corner_size: float,
    right: bool,
    top: bool,
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
            corner_point=corner_point, corner_size=corner_size, right=right, top=top
        )

    return _make_radius(
        corner_point=corner_point,
        corner_size=corner_size,
        n_r=n_r,
        right=right,
        top=top,
    )


def _make_weld(
    *, corner_point: tuple[float, float], corner_size: float, right: bool, top: bool
):
    """
    Create coordinates for a corner weld.

    Parameters
    ----------
    corner_point : tuple
        The original corner point (x, y).
    corner_size : float
        The leg length of the weld.
    right : bool
        Is the weld on the RHS of the web?
    top : bool
        Is the weld at the top of the section.

    Returns
    -------
    list[tuple[float, float]]
        A list containing the new corner point coordinates.
    """
    if right and top:
        return [
            (corner_point[0], corner_point[1] - corner_size),
            (corner_point[0] + corner_size, corner_point[1]),
        ]
    if right:
        return [
            (corner_point[0] + corner_size, corner_point[1]),
            (corner_point[0], corner_point[1] + corner_size),
        ]
    if not right and top:
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
    right: bool,
    top: bool,
    n_r: int = 8,
):
    """
    Generate a list of points representing a circle segment radius for a given
    corner based on the specified position (right, top).

    Parameters
    ----------
    corner_point :
        The original corner point.
    corner_size : float
        The size of the radius
    right : bool
        Indicates if the corner is on the right side.
    top : bool
        Indicates if the corner is on the top side.
    n_r : int
        The number of points around the radius

    Returns
    -------
    list[tuple[float, float]]
        A list containing the new corner point coordinates.
    """

    if right and top:
        corner_point = (
            corner_point[0] + corner_size,
            corner_point[1] - corner_size,
        )
        limit_angles = (90, 180)
    elif right and not top:
        corner_point = (
            corner_point[0] + corner_size,
            corner_point[1] + corner_size,
        )
        limit_angles = (180, 270)
    elif not right and top:
        corner_point = (
            corner_point[0] - corner_size,
            corner_point[1] - corner_size,
        )
        limit_angles = (0, 90)
    else:
        corner_point = (
            corner_point[0] - corner_size,
            corner_point[1] + corner_size,
        )
        limit_angles = (270, 360)

    radius = build_circle(
        centroid=corner_point,
        radius=corner_size,
        no_points=n_r,
        limit_angles=limit_angles,
        use_radians=False,
    )

    return list(reversed(radius))
