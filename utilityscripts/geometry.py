"""
Intended to contain some helper functions & classes for basic geometric operations (calculating areas etc.)
"""

from math import cos, sin, asin, acos, sqrt


class Chord:
    """
    A class to calculate the properties of the chord of a circle.
    """

    def __init__(self, *, r, theta=None, x=None, y2=None):

        self._r = r

        none_list = [theta, x, y2]

        if sum(x is not None for x in none_list) == 0:
            raise Exception("Need to specify one of theta, x or y2.")

        if sum(x is not None for x in none_list) > 1:
            raise Exception("Specify only one of theta, x or y2 need to be specified.")

        self._theta = None
        self._x = None
        self._y2 = None

        if theta is not None:

            self._theta = theta

        elif x is not None:

            self._x = x

        else:

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

        return self.r - self.r * cos(self.theta / 2) if self._y2 is None else self._y2

    @property
    def x(self):

        return 2 * self.r * sin(self.theta / 2) if self._x is None else self._x


if __name__ == "__main__":

    c = Chord(r=1.888, y2=0.1)
    print(c.theta)

    c2 = Chord(r=1.888, theta=c.theta)
    print(c2.y2)
