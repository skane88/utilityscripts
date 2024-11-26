"""
File to contain functions based on Australian Standard AS4100.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from sectionproperties.pre.geometry import Geometry


class CornerDetail(Enum):
    WELD = "weld"
    RADIUS = "radius"


class AS4100Section(ABC):
    """
    Store a cross section that provides AS4100 compatible section properties.

    This is an abstract class,
    and requires implementation for each section type, e.g. I, C, SHS etc.
    """

    def __init__(self, *, section_name):
        self._section_name = section_name

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
    @abstractmethod
    def geometry(self) -> Geometry:
        """
        A sectionproperties Geometry object describing the shape.
        For complex sections can be a CompoundGeometry object
        as this is a sub-class of Geometry.

        Returns
        -------
        Geometry
        """
        pass

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
    def area_gross(self):
        pass

    @property
    @abstractmethod
    def area_net(self):
        pass


class ISection(AS4100Section):
    def __init__(
        self,
        *,
        section_name,
        b_f: float | tuple[float, float],
        d: float,
        t_f: float | tuple[float, float],
        t_w: float,
        corner_detail: CornerDetail | None = None,
        corner_size: float = 0.0,
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
        """

        super().__init__(section_name=section_name)

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
    def geometry(self) -> Geometry:
        raise NotImplementedError()

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
