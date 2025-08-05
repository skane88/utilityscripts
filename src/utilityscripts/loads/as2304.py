"""
Python code for liquid retaining tanks to AS2304
"""

from math import cosh, pi, sinh, tanh

import numpy as np

GAMMA_L_WATER = 10.0  # nominal density of water


class Tank:
    def __init__(
        self,
        diameter: float,
        height: float,
        freeboard: float,
        w_shell: float,
        x_shell: float,
        w_roof: float,
        x_roof: float,
        gamma_l: float = GAMMA_L_WATER,
    ):
        """
        Initialise a tank.

        Parameters
        ----------
        diameter : float
            The diameter of the tank. In m.
        height : float
            The height of the tank. In m.
        freeboard : float
            The freeboard of the tank. In m.
        w_shell : float
            The weight of the shell. In kN.
        x_shell : float
            The height of the shell's COG. In m.
        w_roof : float
            The weight of the roof. In kN.
        x_roof : float
            The height of the roof's COG. In m.

        gamma_l : float, optional
            The weight density of the water, in kN/m^3. By default, GAMMA_L_WATER = 10.0kN/m^3.
        """

        self._diameter = diameter
        self._height = height
        self._freeboard = freeboard
        self._w_shell = w_shell
        self._x_shell = x_shell
        self._w_roof = w_roof
        self._x_roof = x_roof
        self._gamma_l = gamma_l

    @property
    def diameter(self):
        return self._diameter

    @property
    def d(self):
        """
        Diameter of the tank.

        Notes
        -----
        - An alias for self.diameter.
        """

        return self.diameter

    @property
    def radius(self):
        """
        The radius of the tank.
        """

        return self.diameter / 2

    @property
    def height(self):
        """
        The height of the tank. In m.
        """

        return self._height

    @property
    def freeboard(self):
        """
        The freeboard of the tank. In m.
        """

        return self._freeboard

    @property
    def w_shell(self):
        """
        The weight of the shell. In kN.
        """

        return self._w_shell

    @property
    def x_shell(self):
        """
        The height of the shell's COG. In m.
        """

        return self._x_shell

    @property
    def t_shell(self):
        return self._t_shell

    @property
    def w_roof(self):
        """
        The weight of the roof. In kN.
        """

        return self._w_roof

    @property
    def x_roof(self):
        """
        The height of the roof's COG. In m.
        """

        return self._x_roof

    @property
    def gamma_l(self):
        """
        The weight density of the liquid. In kN/m^3.
        """

        return self._gamma_l

    @property
    def h_w(self):
        """
        The height to the liquid's free surface. In m.
        """

        return self.height - self.freeboard

    @property
    def perimeter(self) -> float:
        """
        The perimeter of the tank. In m.
        """

        return pi * self.d

    @property
    def i_xx_force(self):
        """
        The section moment of inertia about the x-axis. In m^3.

        Notes
        -----
        - This is the unit force moment of inertia, similar to a weld moment of inertia.
          Use it to determine forces, not stresses.
        """

        return (pi / 8) * self.d**3

    @property
    def area(self):
        """
        The tank floor area. In m^2.
        """

        return pi * self.radius**2

    @property
    def v_total(self):
        """
        The total volume. In m^3.

        Notes
        -----
        - If freeboard is not 0.0 this will be different from the liquid volume, v.
        """

        return self.height * self.area

    @property
    def v(self):
        """
        The liquid volume after accounting for any freeboard. In m^3.
        """

        return self.h_w * self.area

    @property
    def w_t(self):
        """
        The weight of the tank contents. In kN.
        """

        return self.v * self.gamma_l

    def p_hydrostatic_liquid(self, *, y: float) -> float:
        """
        Calculate the hydrostatic pressure in the tank liquid.

        Parameters
        ----------
        y : float
            The height at which the pressure is considered, taken from the surface of the liquid.
            In m.

        Returns
        -------
        float
            The hydrostatic pressure in the liquid. In kN/m^2 / kPa provided units are met.
        """

        return self.gamma_l * y

    def hs_hydrostatic(self, y: float) -> float:
        """
        Calculate the hydrostatic hoop force (stress) in the tank wall.

        Notes
        -----
        - This is a force / length, not a true stress.
        - To get the hoop stress divide by the wall thickness.

        Parameters
        ----------
        y : float
            The height at which the pressure is considered, taken from the surface of the liquid.
            In m.

        Returns
        -------
        float
            The hydrostatic pressure in the tank wall. In kN/m provided units are met.
        """

        if y > self.h_w:
            raise ValueError(
                "y must be within the tank. "
                + f"{y=:.3e} > {self.h_w=:.3e}, implying the point is below the tank bottom"
            )

        if y < 0:
            raise ValueError(
                "y must be positive. "
                + f"{y=:.3e} < 0, implying the point is above the water level"
            )

        return self.radius * self.p_hydrostatic_liquid(y=y)

    @property
    def alpha_1(self):
        """
        The portion of the liquid that acts inertially in an earthquake.

        Notes
        -----
        - using alpha_1 as the standard does not have a specific symbol
        """

        return s4_6_2_1_alpha_1(d=self.d, h_w=self.h_w)

    @property
    def alpha_2(self):
        """
        The portion of the liquid that acts in a convective fashion during an earthquake.

        Notes
        -----
        - Using alpha_2 as the standard does not have a specific symbol
        """

        return s4_6_2_1_alpha_2(d=self.d, h_w=self.h_w)

    @property
    def x_1_ratio(self):
        """
        The height of the liquid inertial action as a ratio of h_w.
        """

        return s4_6_2_1_x1_ratio(d=self.d, h_w=self.h_w)

    @property
    def x_1(self):
        """
        The height of the liquid inertial action.
        """

        return self.x_1_ratio * self.h_w

    @property
    def x_2_ratio(self):
        """
        The height of the liquid convective action as a ratio of h_w.
        """

        return s4_6_2_1_x2_ratio(d=self.d, h_w=self.h_w)

    @property
    def x_2(self):
        """
        The height of the liquid convective action.
        """

        return self.x_2_ratio * self.h_w

    @property
    def w_1(self):
        """
        The portion of the liquid contents that moves in an inertial fashion.
        """

        return self.w_t * self.alpha_1

    @property
    def w_2(self):
        """
        The portion of the liquid contents that moves in a convective fashion.
        """

        return self.w_t * self.alpha_2

    @property
    def w_i(self):
        """
        The total inertial mass of the tank including liquid and structure
        """

        return self.w_1 + self.w_shell + self.w_roof

    @property
    def w_c(self):
        """
        The total convective mass of the tank including liquid and structure

        Notes
        -----
        - As only liquid behaves in a convective manner, this is the same as w_2.
        """

        return self.w_2

    @property
    def k(self):
        """
        The period factor for earthquake actions.
        """

        return s4_6_2_1_k(d=self.d, h_w=self.h_w)

    @property
    def t_w(self):
        """
        The sloshing wave period, in s.
        """

        return s4_6_2_1_tw(d=self.d, h_w=self.h_w)

    @property
    def c_1(self):
        """
        The sloshing coefficient.
        """

        return s4_6_2_1_c1(d=self.d, h_w=self.h_w)

    def v_b(self, *, k_p: float, z: float, s: float):
        """
        Calculate the base shear due to earthquake actions.

        Notes
        -----
        - The earthquake load is calculated in a manner similar to AS1170.4-1993, but with some modifications.

        Parameters
        ----------
        k_p : float
            Probability factors for earthquake events as per AS1170.4
        z : float
            The acceleration coefficient or hazard design factor as per AS1170.4
        s : float
            A site factor as per AS2304.
            This is similar to the different soil categories in AS1170.4
        """

        return self._v_b1(k_p=k_p, z=z) + self._v_b2(k_p=k_p, z=z, s=s)

    def _v_b1(self, *, k_p: float, z: float):
        """
        Calculate the base shear due to inertial earthquake actions.

        Notes
        -----
        - The earthquake load is calculated in a manner similar to AS1170.4-1993, but with some modifications.

        Parameters
        ----------
        k_p : float
            Probability factors for earthquake events as per AS1170.4
        z : float
            The acceleration coefficient or hazard design factor as per AS1170.4
        """

        acc = 4 * k_p * z
        acc_i = acc * 0.14

        return acc_i * self.w_i

    def _v_b2(self, *, k_p: float, z: float, s: float):
        """
        Calculate the base shear due to convective earthquake actions.

        Notes
        -----
        - The earthquake load is calculated in a manner similar to AS1170.4-1993, but with some modifications.

        Parameters
        ----------
        k_p : float
            Probability factors for earthquake events as per AS1170.4
        z : float
            The acceleration coefficient or hazard design factor as per AS1170.4
        s : float
            A site factor as per AS2304.
            This is similar to the different soil categories in AS1170.4
        """

        acc = 4 * k_p * z
        acc_c = acc * s * self.c_1

        return acc_c * self.w_c

    def m_b(self, *, k_p: float, z: float, s: float) -> float:
        """
        Calculate the base overturning moment due to earthquake actions.

        Notes
        -----
        - The earthquake load is calculated in a manner similar to AS1170.4-1993, but with some modifications.

        Parameters
        ----------
        k_p : float
            Probability factors for earthquake events as per AS1170.4
        z : float
            The acceleration coefficient or hazard design factor as per AS1170.4
        s : float
            A site factor as per AS2304.
            This is similar to the different soil categories in AS1170.4
        """

        return self._m_bi(k_p=k_p, z=z) + self._m_bc(k_p=k_p, z=z, s=s)

    def _m_bi(self, *, k_p: float, z: float) -> float:
        """
        Calculate the base overturning moment due to inertial earthquake actions.

        Notes
        -----
        - The earthquake load is calculated in a manner similar to AS1170.4-1993, but with some modifications.

        Parameters
        ----------
        k_p : float
            Probability factors for earthquake events as per AS1170.4
        z : float
            The acceleration coefficient or hazard design factor as per AS1170.4
        """

        acc = 4 * k_p * z
        acc_i = acc * 0.14

        return acc_i * (
            self.w_shell * self.x_shell
            + self.w_roof * self.x_roof
            + self.w_1 * self.x_1
        )

    def _m_bc(self, *, k_p: float, z: float, s: float) -> float:
        """
        Calculate the base overturning moment due to convective earthquake actions.

        Notes
        -----
        - The earthquake load is calculated in a manner similar to AS1170.4-1993, but with some modifications.

        Parameters
        ----------
        k_p : float
            Probability factors for earthquake events as per AS1170.4
        z : float
            The acceleration coefficient or hazard design factor as per AS1170.4
        s : float
            A site factor as per AS2304.
            This is similar to the different soil categories in AS1170.4
        """

        acc = 4 * k_p * z
        acc_c = acc * s * self.c_1

        return acc_c * self.w_2 * self.x_2

    def s4_6_3_p_1(self, *, y: float, k_p: float, z: float):
        """
        The additional hoop tension due to inertial loads from the liquid in an earthquake.

        Parameters
        ----------
        y : float
            The height at which the pressure is considered. Taken from the surface of the liquid.
            In m.
        k_p : float
            The probability factor for earthquake events as per AS1170.4
        z : float
            The acceleration coefficient or hazard design factor as per AS1170.4

        Returns
        -------
        float
            The total additional hoop tension due to earthquake inertial loads.
            In kN/m provided units are met.
        """

        return s4_6_3_p1(
            y=y, d=self.d, h_w=self.h_w, k_p=k_p, z=z, gamma_l=self.gamma_l
        )

    def s4_6_3_p_2(self, *, y: float, k_p: float, z: float, s: float):
        """
        The total additional hoop tension due convective loads from the liquid in an earthquake.

        Parameters
        ----------
        y : float
            The height at which the pressure is considered. Taken from the surface of the liquid.
            In m.
        k_p : float
            The probability factor for earthquake events as per AS1170.4
        z : float
            The acceleration coefficient or hazard design factor as per AS1170.4
        s : float
            The site factor as per AS2304.

        Returns
        -------
        float
            The total additional hoop tension due to earthquake convective loads.
            In kN/m provided units are met.
        """

        return s4_6_3_p2(
            y=y, d=self.d, h_w=self.h_w, kp=k_p, z=z, s=s, gamma_l=self.gamma_l
        )

    def s4_6_3_p_t(self, *, y: float, k_p: float, z: float, s: float):
        """
        The total additional hoop tension due to earthquake loads.

        Parameters
        ----------
        y : float
            The height at which the pressure is considered. Taken from the surface of the liquid.
            In m.
        k_p : float
            The probability factor for earthquake events as per AS1170.4
        z : float
            The acceleration coefficient or hazard design factor as per AS1170.4
        s : float
            The site factor as per AS2304.

        Returns
        -------
        float
            The total additional hoop tension due to earthquake loads.
            In kN/m provided units are met.
        """

        p1 = self.s4_6_3_p_1(y=y, k_p=k_p, z=z)
        p2 = self.s4_6_3_p_2(y=y, k_p=k_p, z=z, s=s)

        return p1 + p2

    def s5_3_2_1_p_cr_wind(
        self,
        *,
        t: float,
        m: int,
        buckling_length: float | None = None,
        e: float = 200_000_000,
        nu: float = 0.30,
    ) -> float:
        """
        Calculate the critical wind buckling pressure on the tank.

        Notes
        -----
        - The pressure is the external pressure at which buckling occurs, not the internal stresses.
        - This should be iterated until the critical pressure is determined.

        Parameters
        ----------
        m : float, optional
            The number of buckling wavelengths around the tank.
            If None, all m between 1 and max_m are trialled.
        buckling_length : float, optional
            The height of the buckle. Typically the height between stiffeners or the total tank height.
            If None, the total height of the tank is used.
            In m.
        t : float
            The thickness of the tank. In m.
        e : float, optional
            The Young's modulus of the tank. In kPa. For steel this is 200,000,000 kPa
        nu : float, optional
            Poisson's ratio of the tank. For steel this is 0.30.
        max_m : int, optional
            The maximum no. of wavelengths to consider. By default the first 1000.
        """

        if buckling_length is None:
            buckling_length = self.height

        return s5_3_2_1_p_c_wind_uniform(
            m=m, buckling_length=buckling_length, r=self.radius, t=t, e=e, nu=nu
        )

    def s5_3_2_1_p_cr_space(
        self,
        *,
        t: float,
        buckling_length: float | None = None,
        e: float = 200_000_000,
        nu: float = 0.30,
        max_m: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the critical wind buckling pressure for a range of m values.

        Notes
        -----
        - The pressure is the external pressure at which buckling occurs, not the internal stresses.
        - This calculates the pressure for multiple m values.

        Parameters
        ----------
        buckling_length : float, optional
            The height of the buckle. Typically the height between stiffeners or the total tank height.
            If None, the total height of the tank is used.
            In m.
        t : float
            The thickness of the tank. In m.
        e : float, optional
            The Young's modulus of the tank. In kPa. For steel this is 200,000,000 kPa
        nu : float, optional
            Poisson's ratio of the tank. For steel this is 0.30.
        max_m : int, optional
            The maximum no. of wavelengths to consider. By default the first 1000.
        """

        if buckling_length is None:
            buckling_length = self.height

        m_space = np.asarray(range(0, max_m)) + 1
        p_cr_space = s5_3_2_1_p_c_wind_uniform(
            m=m_space, buckling_length=buckling_length, r=self.radius, t=t, e=e, nu=nu
        )

        return m_space, p_cr_space

    def __repr__(self):
        return (
            f"{self.__class__.__name__} d={self.d:.3f}m, height={self.height:.3f}m, "
            + f"h_w={self.h_w:.3f}m, liquid density: {self.gamma_l}kN/m^3.\n"
            + f"Liquid volume {self.v:.1f}m^3, contents weight: {self.w_t:.1f}kN."
        )


def s4_6_2_1_alpha_1(*, d: float, h_w: float) -> float:
    """
    Calculate the portion of the liquid that acts inertially in an earthquake.

    Parameters
    ----------
    d : float
        The diameter of the tank.
    h_w : float
        The height of the liquid in the tank during the earthquake.
    """

    d_h_w = d / h_w

    if d_h_w >= 4 / 3:
        return tanh(0.866 * d_h_w) / (0.866 * d_h_w)

    return 1.0 - 0.218 * d_h_w


def s4_6_2_1_alpha_2(*, d: float, h_w: float) -> float:
    """
    The portion of the liquid that acts in a convective fashion during an earthquake.

    Parameters
    ----------
    d : float
        The diameter of the tank.
    h_w : float
        The height of the liquid in the tank during the earthquake.
    """

    d_h_w = d / h_w
    return 0.230 * d_h_w * tanh(3.67 / d_h_w)


def s4_6_2_1_x1_ratio(*, d: float, h_w: float) -> float:
    """
    The equivalent height of the liquid inertial action.
    As a ratio of x1 / h_w.

    Parameters
    ----------
    d : float
        The diameter of the tank.
    h_w : float
        The height of the liquid in the tank during the earthquake.
    """

    if d / h_w >= 4 / 3:
        return 0.375

    return 0.5 - 0.094 * (d / h_w)


def s4_6_2_1_x2_ratio(*, d: float, h_w: float) -> float:
    """
    The equivalent height of the liquid convective action.
    As a ratio of x2 / h_w

    Parameters
    ----------
    d : float
        The diameter of the tank.
    h_w : float
        The height of the liquid in the tank during the earthquake.
    """

    d_h_w = d / h_w

    a = cosh(3.67 / d_h_w) - 1.0
    b = (3.67 / d_h_w) * sinh(3.67 / d_h_w)

    return 1.0 - a / b


def s4_6_2_1_k(*, d: float, h_w: float) -> float:
    """
    The sloshing period factor.

    Parameters
    ----------
    d : float
        The diameter of the tank.
    h_w : float
        The height of the liquid in the tank during the earthquake.
    """

    return 0.578 / (tanh(3.67 / (d / h_w))) ** 0.5


def s4_6_2_1_tw(*, d: float, h_w: float) -> float:
    """
    The first mode sloshing period.

    Parameters
    ----------
    d : float
        The diameter of the tank.
    h_w : float
        The height of the liquid in the tank during the earthquake.

    Returns
    -------
    float
        The first mode sloshing period. In s if units for inputs are m.
    """

    return 1.811 * s4_6_2_1_k(d=d, h_w=h_w) * (d**0.5)


def s4_6_2_1_c1(*, d: float, h_w=float) -> float:
    """
    The sloshing coefficient.

    Parameters
    ----------
    d : float
        The diameter of the tank.
    h_w : float
        The height of the liquid in the tank during the earthquake.
    """

    t_w = s4_6_2_1_tw(d=d, h_w=h_w)

    if t_w < 4.5:  # noqa: PLR2004
        return 1 / (6 * t_w)

    return 0.75 / (t_w**2)


def s4_6_3_p1(
    y, *, d: float, h_w: float, k_p: float, z: float, gamma_l: float = GAMMA_L_WATER
) -> float:
    """
    The impulsive pressure in the tank wall due to inertial behaviour of the fluid.

    Parameters
    ----------
    y : float
        The height at which the pressure is considered, taken from the surface of the liquid.
        In m.
    d : float
        The diameter of the tank. In m.
    h_w : float
        The height of the liquid in the tank during the earthquake. In m.
    k_p : float
        The probability factor for earthquake events as per AS1170.4
    z : float
        The acceleration coefficient or hazard design factor as per AS1170.4
    gamma_l : float, optional
        The density of the liquid, by default GAMMA_L_WATER.

    Returns
    -------
    float
        The impulsive pressure in the tank wall due to inertial behaviour of the fluid.
        In kN/m provided units are met.
    """

    if y > h_w:
        raise ValueError(
            "y must be within the tank. "
            + f"{y=:.3e} > {h_w=:.3e}, implying the point is below the tank bottom"
        )

    if y < 0:
        raise ValueError(
            "y must be positive. "
            + f"{y=:.3e} < 0, implying the point is above the water level"
        )

    sg = gamma_l / GAMMA_L_WATER

    d_h_w = d / h_w

    p_base = sg * k_p * z

    if d_h_w > 4 / 3:
        return (
            4.751
            * p_base
            * d
            * h_w
            * ((y / h_w) - 0.5 * (y / h_w) ** 2)
            * tanh(0.866 * d_h_w)
        )

    if y < 0.75 * d:
        return 2.9237 * p_base * d**2 * ((y / (0.75 * d)) - 0.5 * (y / (0.75 * d)) ** 2)

    return 1.4666 * p_base * d**2


def s4_6_3_p2(
    y,
    *,
    d: float,
    h_w: float,
    kp: float,
    z: float,
    s: float,
    gamma_l: float = GAMMA_L_WATER,
) -> float:
    """
    The impulsive pressure in the tank wall due to convective behaviour of the fluid.

    Parameters
    ----------
    y : float
        The height at which the pressure is considered. Taken from the surface of the liquid.
        In m.
    d : float
        The diameter of the tank. In m.
    h_w : float
        The height of the liquid in the tank during the earthquake. In m.
    k_p : float
        The probability factor for earthquake events as per AS1170.4
    z : float
        The acceleration coefficient or hazard design factor as per AS1170.4
    s : float
        The site factor as per AS2304.
    gamma_l : float, optional
        The density of the liquid, by default GAMMA_L_WATER.

    Returns
    -------
    float
        The impulsive pressure in the tank wall due to inertial behaviour of the fluid.
        In kN/m provided units are met.
    """

    if y > h_w:
        raise ValueError(
            "y must be within the tank. "
            + f"{y=:.3e} > {h_w=:.3e}, implying the point is below the tank bottom"
        )

    if y < 0:
        raise ValueError(
            "y must be positive. "
            + f"{y=:.3e} < 0, implying the point is above the water level"
        )

    sg = gamma_l / GAMMA_L_WATER
    c1 = s4_6_2_1_c1(d=d, h_w=h_w)

    return (
        7.3479
        * sg
        * kp
        * z
        * s
        * c1
        * d**2
        * cosh((3.68 * (h_w - y)) / d)
        / cosh(3.68 * h_w / d)
    )


def s5_3_2_1_p_c_wind_uniform(
    *,
    m: float,
    buckling_length: float,
    r: float,
    t: float,
    e: float = 200_000_000,
    nu: float = 0.30,
) -> float:
    """
    Calculate the critical buckling pressure of the tank under external compressive loading.

    Notes
    -----
    - The pressure is the external pressure at which buckling occurs, not the internal stresses.
    - This should be iterated until the critical pressure is determined.

    Parameters
    ----------
    m : float
        The number of buckling wavelengths around the tank.
    buckling_length : float
        The height of the buckle. Typically the height between stiffeners or the total tank height.
        In m.
    r : float
        The radius of the tank. In m.
    t : float
        The thickness of the tank. In m.
    e : float, optional
        The Young's modulus of the tank. In kPa. For steel this is 200,000,000 kPa
    nu : float, optional
        Poisson's ratio of the tank. For steel this is 0.30.
    """

    r = r * 1000
    t = t * 1000
    buckling_length = buckling_length * 1000

    alpha = (pi * r) / (m * buckling_length)
    aa = (e * t**3) / (12 * (1 - nu**2) * r**3)
    bb = (pi * r / buckling_length) ** 2
    cc = (alpha + 1 / alpha) ** 2
    dd = e * t / r
    ee = (buckling_length / (pi * r)) ** 2
    ff = alpha**4 / (alpha + 1 / alpha) ** 2

    return aa * bb * cc + dd * ee * ff


def s5_3_3_lambda(*, r: float, t: float) -> float:
    """
    Calculate the meridional bending half wavelength for a circumferential stiffener.

    Parameters
    ----------
    r : float
        The tank radius. In m.
    t : float
        The tank thickness. In m.
    """

    return 2.4 * (r * t) ** 0.5


def s5_3_3_f_des(*, p_t: float, r: float, h_s: float) -> float:
    """
    Calculate the design force for a circumferential stiffener.

    Parameters
    ----------
    p_t : float
        The design wind pressure on the tank. In kPa.
    r : float
        The tank radius. In m.
    h_s : float
        The effective wall height to include with the stiffener.
    """

    return p_t * r * h_s


def s5_3_3_f_capacity(*, i_z: float, r_r: float, e: float = 200_000_000):
    """
    The elastic buckling capacity of a stiffener.

    Parameters
    ----------
    i_z : float
        The section moment of inertia about the vertical axis, of the stiffener element.
        In m^4.
    r_r : float
        The radius to the centroid of the stiffener. In m.
    e : float
        The Young's modulus of the stiffener. In kPa.

    Returns
    -------
    float
        The buckling capacity of the stiffener, in kN.
    """

    return 3 * e * i_z / (r_r**2)
