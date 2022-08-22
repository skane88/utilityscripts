"""
Intended to contain some helper functions & classes for basic geometric operations (calculating areas etc.)
"""

from math import acos, asin, cos, pi, sin, sqrt


class Chord:
    """
    A class to calculate the properties of the chord of a circle.

    NOTE: a Chord is limited to the case where the chord length is less than
    the diameter of the circle.
    """

    def __init__(self, *, r, theta=None, x=None, y2=None):
        """
        Define a Chord object. A chord object can be fully defined provided
        the radius of the circle (r) and one of the following are provided:
            * The angle of the chord (theta),
            * The length of the chord (x), or
            * The thickness of the chord (y2).

        :param r: The radius of the circle.
        :param theta: The angle of the Chord.
        :param x: The length of the Chord. Must be <= 2*r, which limits a
            Chord defined this way to half a circle. The assumption is
            that the Chord is the smaller resulting Chord of the circle.
        :param y2: The thickness of the Chord.
        """

        self._r = r

        none_list = [theta, x, y2]

        if sum(x is not None for x in none_list) == 0:
            raise ValueError("Need to specify one of theta, x or y2.")

        if sum(x is not None for x in none_list) > 1:
            raise ValueError("Specify only one of theta, x or y2 need to be specified.")

        self._theta = None
        self._x = None
        self._y2 = None

        if theta is not None:

            assert theta <= 2 * pi

            self._theta = theta

        elif x is not None:

            assert x <= 2 * r

            self._x = x

        else:

            assert y2 <= 2 * r

            self._y2 = y2

    @property
    def r(self):
        return self._r

    @property
    def theta(self):

        if self._theta is not None:
            return self._theta

        elif self._y2 is None:
            return 2 * asin(self.x / (2 * self.r))

        else:
            return 4 * asin(sqrt(self.y2 / (2 * self.r)))

    @property
    def y2(self):

        if self._y2 is not None:
            return self._y2

        elif self._theta is not None:
            return self.r - self.r * cos(self.theta / 2)

        else:
            return self.r * (1 - cos(asin(self.x / (2 * self.r))))

    @property
    def x(self):

        if self._x is not None:
            return self._x

        elif self._y2 is not None:
            return 2 * self.r * sin(2 * asin(sqrt(self.y2 / (2 * self.r))))

        else:
            return 2 * self.r * sin(self.theta / 2) if self._x is None else self._x

    @property
    def area(self):

        return ((self.r**2) / 2) * (self.theta - sin(self.theta))

    def point(self, phi):

        if not 0 <= phi <= self.theta:
            raise ValueError("Expected phi to be in the range of 0 <= phi <= theta")

        start_angle = (pi - self.theta) / 2
        angle = start_angle + phi

        return self.r * cos(angle), self.r * sin(angle)


if __name__ == "__main__":

    c1 = Chord(r=1.888, y2=0.1)
    print("Chord c1")
    print(c1.theta)
    print(c1.y2)
    print(c1.x)
    print()

    c2 = Chord(r=1.888, theta=c1.theta)
    print("Chord c2")
    print(c2.y2)
    print()

    c3 = Chord(r=1.888, theta=2 * pi)
    print("Chord c3")
    print(c3.theta)
    print(c3.y2)
    print(c3.x)
    print()

    c4 = Chord(r=1.888, y2=2 * 1.888 - 0.1)
    print("Chord c4")
    print(c4.theta)
    print(c4.y2)
    print(c4.x)
    print()
