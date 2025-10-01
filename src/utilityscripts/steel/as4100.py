"""
File to contain functions based on Australian Standard AS4100.
"""

from __future__ import annotations

from enum import Enum

from utilityscripts.geometry import Circle
from utilityscripts.steel.steel import bolt_grades

PHI_STEEL = {"φ_s": 0.90, "φ_w.sp": 0.80, "φ_w.gp": 0.60}
PHI_STEEL["steel"] = PHI_STEEL["φ_s"]
PHI_STEEL["weld, sp"] = PHI_STEEL["φ_w.sp"]
PHI_STEEL["weld, gp"] = PHI_STEEL["φ_w.gp"]


class RestraintCode(Enum):
    """
    The bending restraint code for the section.
    """

    F = "F"
    L = "L"
    P = "P"
    U = "U"
    C = "C"  # continuous


class WebType(Enum):
    STIFFENED = "stiffened"
    UNSTIFFENED = "unstiffened"


def s5_6_1_1_alpha_m(*, m_m, m_2, m_3, m_4):
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


def s5_6_3_k_t(
    *,
    restraint_code: tuple[RestraintCode, RestraintCode] | str,
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
        The number of webs in the beam section.

    Returns
    -------
    float
        Computed twist restraint factor.

    Examples
    --------
    >>> s5_6_3_k_t(restraint_code="PP", d_1=0.206, length=4.0, t_f=0.012, t_w=0.0065)
    1.08101

    >>> s5_6_3_k_t(restraint_code="FF", d_1=0.206, length=4.0, t_f=0.012, t_w=0.0065)
    1.00000
    """

    if isinstance(restraint_code, str):
        restraint_code = (
            RestraintCode(restraint_code[0]),
            RestraintCode(restraint_code[1]),
        )

    if restraint_code in [
        (RestraintCode.F, RestraintCode.F),
        (RestraintCode.F, RestraintCode.L),
        (RestraintCode.L, RestraintCode.F),
        (RestraintCode.L, RestraintCode.L),
        (RestraintCode.F, RestraintCode.U),
        (RestraintCode.U, RestraintCode.F),
    ]:
        return 1.0

    if restraint_code in [
        (RestraintCode.F, RestraintCode.P),
        (RestraintCode.P, RestraintCode.F),
        (RestraintCode.P, RestraintCode.L),
        (RestraintCode.L, RestraintCode.P),
        (RestraintCode.P, RestraintCode.U),
        (RestraintCode.U, RestraintCode.P),
    ]:
        return 1 + ((d_1 / length) * (t_f / (2 * t_w)) ** 3) / n_w

    return 1 + (2.0 * (d_1 / length) * (t_f / (2 * t_w)) ** 3) / n_w


def s5_11_5_alpha_v(
    *,
    d_p,
    t_w,
    f_y,
    s: float | None = None,
    f_y_ref=250.0,
    web_type: WebType = WebType.UNSTIFFENED,
):
    """
    Calculate the stiffened web shear buckling parameter alpha_v as per
    AS4100 S5.11.5.1 and S5.11.5.2.

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
    s : float | None, optional
        The length of the web or spacing between vertical stiffeners
        that meet the requirements of AS4100.
        If web is not stiffened this value is ignored.
    f_y_ref : float, optional
        The reference yield stress, nominally 250.0.
        Default is 250.0MPa.
    web_type : WebType, optional
        The type of web. If STIFFENED, use S5.11.5.2, otherwise use S5.11.5.1.
        Default is WebType.UNSTIFFENED.

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
    >>> web_type = WebType.STIFFENED
    >>> s5_11_5_2_alpha_v(d_p=d_p, t_w=t_w, f_y=f_y, s=s, web_type=web_type)
    0.918947
    """

    a1 = (82 / ((d_p / t_w) * (f_y / f_y_ref) ** 0.5)) ** 2

    if web_type == WebType.UNSTIFFENED:
        return min(a1, 1.0)

    a2_param = 1.0 if (s / d_p) <= 1.0 else 0.75

    a2 = (a2_param / ((s / d_p) ** 2)) + 1.0

    return min(a1 * a2, 1.0)


def s5_11_5_2_alpha_d(*, alpha_v, d_p, s):
    """
    Calculate the stiffened web shear buckling parameter alpha_d as per
    AS4100 S5.11.5.2.

    Does not check for the presence of a stiffened end post.

    Parameters
    ----------
    alpha_v : float
        The stiffened web shear buckling parameter alpha_v for STIFFENED webs as per
        AS4100 S5.11.5.2.
    d_p : float
        The depth of the web panel.
    s : float
        The length of the web or spacing of vertical stiffeners that meet the
        requirement of AS4100.

    Returns
    -------
    float
        The stiffened web shear buckling parameter alpha_d.
    """

    return 1 + ((1 - alpha_v) / (1.15 * alpha_v * (1 + (s / d_p) ** 2) ** 0.5))


def s5_15_3_a_s(*, gamma, a_w, alpha_v, v_star, v_u, d_p, s, phi=0.9):
    """
    Calculate the minimum area of a transverse shear stiffener as per AS4100 S5.15.3

    Parameters
    ----------
    gamma : float
        The stiffener type factor.
    a_w : float
        The area of the web.
    alpha_v : float
        The stiffened web shear buckling parameter alpha_v as per
        AS4100 S5.11.5.2.
    v_star : float
        The shear in the web.
    v_u : float
        The ultimate shear capacity of the web.
    d_p : float
        The depth of the web panel.
    s : float
        The length of the web or spacing of vertical stiffeners that meet the
        requirement of AS4100.
    phi : float, optional
        The capacity reduction factor.
        Default is 0.9.

    Returns
    -------
    float
        The minimum area of a transverse shear stiffener.
    """

    a_1 = 0.5 * gamma * a_w
    a_2 = 1 - alpha_v
    a_3 = v_star / (phi * v_u)
    a_4 = (s / d_p) - ((s / d_p) ** 2 / (1 + (s / d_p) ** 2) ** 0.5)

    return a_1 * a_2 * a_3 * a_4


def s9_1_9_block(
    *,
    a_gv: float,
    a_nv: float,
    a_nt: float,
    f_yp: float,
    f_up: float,
    k_bs: float,
):
    """
    Calculate the block shear capacity as per AS4100.

    Parameters
    ----------
    a_gv : float
        The gross shear failure area.
    a_nv : float
        The net shear failure area.
    a_nt : float
        The net tension failure area.
    f_yp : float
        The yield strength of the plate.
    f_up : float
        The ultimate strength of the plate.
    k_bs : float
        The block shear eccentrictity factor.
    """

    return min(0.6 * a_nv * f_up, 0.6 * a_gv * f_yp) + k_bs * a_nt * f_up


class Bolt:
    def __init__(
        self,
        *,
        d_f: float,
        grade: str,
        stresses: tuple[float, float] | None = None,
        a_c: float | None = None,
        a_o: float | None = None,
        a_s: float | None = None,
    ):
        """
        Class to represent a bolt.

        Parameters
        ----------
        d_f : float
            The bolt diameter. In m.
        grade : str
            The grade of the bolt, e.g. 4.6, 8.8 etc.
        stresses : tuple[float, float] | None
            The nominal tensile stress and the ultimate tensile stress of the bolt:
            (f_yf, f_uf) in Pa.
            If None is provided, it is calculated based on the grade and the rules in
            AS1275. If the grade is not found in the grade data an error will be raised.
        a_c : float | None
            The minor diameter area of the bolt. In m². If None is provided, it is
            calculated based on the diameter and the rules in AS1275.
        a_o : float | None
            The nominal plain shank area of the bolt. In m². If None is provided, it is
            calculated based on the diameter and the rules in AS1275.
        a_s : float | None
            The tensile stress area of the bolt. In m². If None is provided, it is
            calculated based on the diameter and the rules in AS1275.
        """

        self._d_f = d_f
        self._grade = grade

        self._f_yf = 0.0
        self._f_uf = 0.0

        if stresses is None:
            bolt_grade = bolt_grades()[grade]
            self._f_yf = bolt_grade.get_f_yf(d_f)
            self._f_uf = bolt_grade.get_f_uf(d_f)

        else:
            self._f_yf = stresses[0]
            self._f_uf = stresses[1]

        # TODO: set the a_c, a_o and a_s values.

        if a_c is None:
            self._a_c = 0.0

        if a_o is None:
            self._a_o = 0.0

        if a_s is None:
            self._a_s = 0.0

        self._a_c = a_c
        self._a_o = a_o
        self._a_s = a_s

    @property
    def d_f(self) -> float:
        """
        The diameter of the bolt. In m.
        """

        return self._d_f

    @property
    def grade(self) -> str | None:
        """
        The grade of the bolt. If the bolt object was set up with an f_uf / f_yf tuple,
        this will return None.
        """

        return self._grade

    @property
    def a_c(self) -> float:
        """
        The minor diameter area of the bolt. In m².
        """

        return self._a_c

    @property
    def a_o(self) -> float:
        """
        The nominal plain shank area of the bolt. In m².
        """

        return self._a_o

    @property
    def a_s(self) -> float:
        """
        The tensile stress area of the bolt. In m².
        """

        return self._a_s


def s9_2_2_4_v_by(*, d_f: float, t_p: float, f_up: float) -> float:
    """
    Calculate the bolt hole bearing capacity, as limited by local yielding, as per
    AS4100 S9.2.2.4

    Parameters
    ----------
    d_f : float
        The fastener diameter.
    t_p : float
        The thickness of the plate.
    f_up : float
        The ultimate tensile stress of the plate.
    """

    return 3.2 * d_f * t_p * f_up


def s9_2_2_4_v_bt(*, a_e: float, t_p: float, f_up: float) -> float:
    """
    Calculate the bolt hole bearing capacity, as limited by tear-out, as per
    AS4100 S9.2.2.4

    Parameters
    ----------
    a_e : float
        Fastener edge distance. In m.
    t_p : float
        Thickness of plate. In m.
    f_up : float
        The ultimate tensile stress of the plate. In Pa.

    Returns
    -------
    float
        The bolt hole bearing capacity, as limited by tear-out, in N.
    """

    return a_e * t_p * f_up


def s9_4_1_v_f(*, d_pin: float, n_s: float, f_yp: float) -> float:
    """
    Calculate the capacity of a pin in shear as per AS4100 S9.4.1

    Parameters
    ----------
    d_pin : float
        The diameter of the pin. In m.
    n_s : float
        The number of shear planes.
    f_yp : float
        The yield strength of the pin. In Pa.

    Returns
    -------
    float
        The capacity of the pin in shear. In N.
    """

    circle = Circle(d=d_pin)
    return 0.62 * n_s * circle.area * f_yp


def s9_4_2_v_b(*, d_pin: float, t_p: float, f_yp: float, k_p: float) -> float:
    """
    Calculate the capacity of a pin in bearing as per AS4100 S9.4.2

    Parameters
    ----------
    d_pin : float
        The diameter of the pin. In m.
    t_p : float
        The thickness of the plate. In m.
    f_yp : float
        The yield strength of the plate. In Pa.
    k_p : float
        The rotation factor.
        Typically 1.0 for stationary pins and 0.5 for rotating pins.

    Returns
    -------
    float
        The capacity of the pin in bearing. In N.
    """

    return 1.4 * f_yp * t_p * d_pin * k_p


def s9_4_3_m_p(*, d_pin: float, f_yp: float) -> float:
    """
    Calculate the capacity of a pin in bending as per AS4100 S9.4.3

    Parameters
    ----------
    d_pin : float
        The diameter of the pin. In m.
    f_yp : float
        The yield strength of the pin. In Pa.

    Returns
    -------
    float
        The capacity of the pin in bending. In Nm.
    """

    s = d_pin**3 / 6

    return f_yp * s


def s9_6_3_10_v_w(
    *, size: float, f_uw: float, k_r: float = 1.0, phi_weld: float = 0.8
) -> float:
    """
    Calculate the capacity of a fillet weld as per AS4100-2020

    Parameters
    ----------
    size : float
        The leg length of the weld. In m.
        Future versions of this function may be extended to
        allow for uneven legs.
    f_uw : float
        The ultimate strength of the weld metal. In Pa.
    k_r : float
        The weld length parameter. For welds less than
        1.7m long (almost all welds) this is 1.00.
    phi_weld : float
        The capacity reduction factor.

    Returns
    -------
    float
        The capacity of the weld in N/m.
    """

    t_t = size / (2**0.5)
    return phi_weld * 0.60 * t_t * k_r * f_uw
