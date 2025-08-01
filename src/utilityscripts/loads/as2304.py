"""
Python code for liquid retaining tanks to AS2304
"""

from math import cosh, pi, sinh, tanh


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
        gamma_l: float = 10,
    ):
        """
        Initialise a tank.

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
        return self._height

    @property
    def freeboard(self):
        return self._freeboard

    @property
    def w_shell(self):
        return self._w_shell

    @property
    def x_shell(self):
        return self._x_shell

    @property
    def w_roof(self):
        return self._w_roof

    @property
    def x_roof(self):
        return self._x_roof

    @property
    def gamma_l(self):
        """
        The weight density of the liquid.
        """

        return self._gamma_l

    @property
    def h_w(self):
        return self.height - self.freeboard

    @property
    def area(self):
        return pi * self.radius**2

    @property
    def v_total(self):
        return self.height * self.area

    @property
    def v(self):
        """
        The liquid volume after accounting for any freeboard.
        """

        return self.h_w * self.area

    @property
    def w_t(self):
        """
        The weight of the tank contents.
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

        if self.d / self.h_w >= 4 / 3:
            return tanh(0.866 * self.d / self.h_w) / (0.866 * self.d / self.h_w)

        return 1.0 - 0.218 * self.d / self.h_w

    @property
    def alpha_2(self):
        """
        The portion of the liquid that acts in a convective fashion during an earthquake.

        Notes
        -----
        - Using alpha_2 as the standard does not have a specific symbol
        """

        d_h_w = self.d / self.h_w

        return 0.230 * d_h_w * tanh(3.67 / d_h_w)

    @property
    def x_1(self):
        """
        The height of the liquid inertial action.
        """

        if self.d / self.h_w >= 4 / 3:
            return 0.375 * self.h_w

        return (0.5 - 0.094 * (self.d / self.h_w)) * self.h_w

    @property
    def x_2(self):
        """
        The height of the liquid convective action.
        """

        d_h_w = self.d / self.h_w

        a = cosh(3.67 / d_h_w) - 1.0
        b = (3.67 / d_h_w) * sinh(3.67 / d_h_w)

        return (1.0 - a / b) * self.h_w

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

        return 0.578 / (tanh(3.67 / (self.d / self.h_w))) ** 0.5

    @property
    def t_w(self):
        """
        The sloshing wave period, in s.
        """

        return 1.811 * self.k * (self.diameter**0.5)

    @property
    def c_1(self):
        """
        The sloshing coefficient.
        """

        if self.t_w < 4.5:  # noqa: PLR2004
            return 1 / (6 * self.t_w)

        return 0.75 / (self.t_w**2)

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
