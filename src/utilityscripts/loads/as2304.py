"""
Python code for liquid retaining tanks to AS2304
"""

from math import cosh, pi, sinh, tanh

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

        acc = 4 * k_p * z
        acc_i = acc * 0.14
        acc_c = acc * s * self.c_1

        return acc_i * self.w_i + acc_c * self.w_c

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

        acc = 4 * k_p * z
        acc_i = acc * 0.14
        acc_c = acc * s * self.c_1

        m_i = acc_i * (
            self.w_shell * self.x_shell
            + self.w_roof * self.x_roof
            + self.w_1 * self.x_1
        )
        m_c = acc_c * self.w_2 * self.x_2

        return m_i + m_c

    def s4_6_3_p_1(self, *, y: float, k_p: float, z: float):
        return s4_6_3_p1(
            y=y, d=self.d, h_w=self.h_w, k_p=k_p, z=z, gamma_l=self.gamma_l
        )

    def s4_6_3_p_2(self, *, y: float, k_p: float, z: float, s: float):
        return s4_6_3_p2(
            y=y, d=self.d, h_w=self.h_w, kp=k_p, z=z, s=s, gamma_l=self.gamma_l
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
