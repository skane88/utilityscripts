"""
Equations of linear motion.
"""


class Linear:
    def __init__(
        self,
        *,
        v: float | None = None,
        v_0: float | None = None,
        s: float | None = None,
        s_0: float | None = None,
        a: float | None = None,
        t: float | None = None,
    ):
        self._v = v
        self._v_0 = v_0
        self._s = s
        self._s_0 = s_0
        self._a = a
        self._t = t

        self._calculate()

    def _calculate(self):
        """
        Iteratively calculate as many properties as possible.
        """

        i = 0

        while i < 5:  # noqa: PLR2004
            # assume 5 iterations is enough because there are 5x equations of motion.

            if all(
                val is not None
                for val in (self._v, self._v_0, self._s, self._s_0, self._a, self._t)
            ):
                # if we know everything, break
                break

            # now go through the equations of linear motion and solve for as many
            # as we possibly can.
            if self._v is None:
                self._calculate_v()

            if self._v_0 is None:
                self._calculate_v_0()

            if self._s is None:
                self._calculate_s()

            if self._s_0 is None:
                self._calculate_s_0()

            if self._a is None:
                self._calculate_a()

            if self._t is None:
                self._calculate_t()

            i += 1

    def _calculate_v(self):
        """
        Helper method to calculate v if possible given the available data
        """

        if all(val is not None for val in (self._v_0, self._a, self._t)):
            self._v = self._v_0 + self._a * self._t
        elif all(val is not None for val in (self._s, self._s_0, self._v_0, self._t)):
            self._v = (self._s - self._s_0) * 2 / self._t - self._v_0
        elif all(val is not None for val in (self._v_0, self._a, self._s, self._s_0)):
            self._v = (self._v_0**2 + 2 * self._a * (self._s - self._s_0)) ** 0.5
        elif all(val is not None for val in (self._s, self._s_0, self._t, self._a)):
            self._v = (self._s - self._s_0 + 0.5 * self._a * self._t**2) / self._t

    def _calculate_v_0(self):
        """
        Helper method to calculate v_0 if possible given the available data
        """

        if all(val is not None for val in (self._v, self._a, self._t)):
            self._v_0 = self._v - self._a * self._t
        elif all(val is not None for val in (self._s, self._s_0, self._t, self._a)):
            self._v_0 = (self._s - self._s_0 - 0.5 * self._a * self._t**2) / self._t
        elif all(val is not None for val in (self._s, self._s_0, self._v, self._t)):
            self._v_0 = 2 * (self._s - self._s_0) / self._t - self._v
        elif all(val is not None for val in (self._v, self._a, self._s, self._s_0)):
            self._v_0 = (self._v**2 - 2 * self._a * (self._s - self._s_0)) ** 0.5

    def _calculate_s(self):
        """
        Helper method to calculate s if possible given the available data
        """

        if all(val is not None for val in (self._s_0, self._v_0, self._a, self._t)):
            self._s = self._s_0 + self._v_0 * self._t + 0.5 * self._a * self._t**2
        elif all(val is not None for val in (self._s_0, self._v, self._v_0, self._t)):
            self._s = self._s_0 + 0.5 * (self._v + self._v_0) * self._t

            # TODO: implement remaining s method

        elif all(val is not None for val in (self._s_0, self._v, self._t, self._a)):
            self._s = self._s_0 + self._v * self._t - 0.5 * self._a * self._t**2

    def _calculate_s_0(self):
        """
        Helper method to calculate s_0 if possible given the available data
        """

        if all(val is not None for val in (self._s, self._v_0, self._a, self._t)):
            self._s_0 = self._s - self._v_0 * self._t - 0.5 * self._a * self._t**2
        elif all(val is not None for val in (self._s, self._v, self._v_0, self._t)):
            self._s_0 = self._s - 0.5 * (self._v + self._v_0) * self._t

            # TODO: implement remaining s_0 method

        elif all(val is not None for val in (self._s, self._v, self._a, self._t)):
            self._s_0 = self._s - self._v * self._t + 0.5 * self._a * self._t**2

    def _calculate_a(self):
        """
        Helper method to calculate a if possible given the available data
        """

        if all(val is not None for val in (self._v, self._v_0, self._t)):
            self._a = (self._v - self._v_0) / self._t
        if all(val is not None for val in (self._s, self._s_0, self._v_0, self._t)):
            self._a = (2 * (self._s - self._s_0 - self._v_0 * self._t)) / self._t**2
        if all(val is not None for val in (self._v, self._v_0, self._s, self._s_0)):
            self._a = (self._v**2 - self._v_0**2) / (2 * (self._s - self._s_0))
        if all(val is not None for val in (self._s, self._s_0, self._v, self._t)):
            self._a = (-2 * (self._s - self._s_0 - self._v * self._t)) / self._t**2

    def _calculate_t(self):
        """
        Helper method to calculate t if possible given the available data
        """

        if all(val is not None for val in (self._v, self._v_0, self._a)):
            self._t = (self._v - self._v_0) / self._a

            # TODO: Implement rest of the t methods

    @property
    def v(self):
        if self._v is None:
            raise ValueError("v is not known")

        return self._v

    @property
    def v_0(self):
        if self._v_0 is None:
            raise ValueError("v_0 is not known")

        return self._v_0

    @property
    def s(self):
        if self._s is None:
            raise ValueError("s is not known")

        return self._s

    @property
    def s_0(self):
        if self._s_0 is None:
            raise ValueError("s_0 is not known")

        return self._s_0

    @property
    def a(self):
        if self._a is None:
            raise ValueError("a is not known")

        return self._a

    @property
    def t(self):
        if self._t is None:
            raise ValueError("t is not known")

        return self._t
