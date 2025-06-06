"""
Intended to contain some helper functions & classes for basic geometric operations
(calculating areas etc.). This is not intended to calculate section properties.
"""

from math import asin, cos, pi, radians, sin, sqrt

import numpy as np
from shapely.geometry import Point


class Circle:
    """
    A class to calculate some basic properties of circles.
    """

    def __init__(self, *, r=None, d=None):
        """
        Define a Circle object. A circle can be defined by either a radius or a
        diameter.

        :param r: the radius of the circle.
        :param d: the diameter of the circle.
        """

        none_list = [r, d]

        self._r = None
        self._d = None

        if sum(x is not None for x in none_list) == 0:
            raise ValueError("Need to specify one of r or d.")

        if sum(x is not None for x in none_list) > 1:
            raise ValueError("Specify only one of tr or d.")

        if r is None:
            self._d = d

        else:
            self._r = r

    @property
    def r(self):
        """
        The radius of the circle.
        """
        return self.d / 2 if self._r is None else self._r

    @property
    def d(self):
        """
        The diameter of the circle.
        """

        return self.r * 2 if self._d is None else self._d

    @property
    def area(self):
        """
        The area of the circle.
        """

        return pi * self.r**2

    @property
    def circumference(self):
        """
        The circumference of the circle.
        """

        return self.d * pi


class CircularSegment:
    """
    A class to calculate the properties of a Segment of a Circle.
    """

    def __init__(self, *, r, theta=None, x=None, y2=None):
        """
        Define a Circular_Segment object. This can be fully defined provided
        the radius of the circle (r) and one of the following properties of
        the chord that defines the segment are provided:
            * The angle of the chord (theta),
            * The length of the chord (x), or
            * The thickness of the chord (y2).

        :param r: The radius of the circle.
        :param theta: The angle of the chord.
        :param x: The length of the chord. Must be <= 2*r, which limits a
            segment defined this way to half a circle. The assumption is
            that the segment is the smaller resulting segment.
        :param y2: The thickness of the segment.
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
            if theta > 2 * pi:
                raise ValueError(f"theta must be <= 2 * pi. Was {theta}")

            self._theta = theta

        elif x is not None:
            if x > 2 * r:
                raise ValueError(f"x must be <= 2*r. Was {x}")

            self._x = x

        else:
            if y2 > 2 * r:
                raise ValueError(f"y2 must be <= 2*r. Was {y2}")

            self._y2 = y2

    @property
    def r(self):
        return self._r

    @property
    def theta(self):
        if self._theta is not None:
            return self._theta

        if self._y2 is None:
            return 2 * asin(self.x / (2 * self.r))

        return 4 * asin(sqrt(self.y2 / (2 * self.r)))

    @property
    def y2(self):
        if self._y2 is not None:
            return self._y2

        if self._theta is not None:
            return self.r - self.r * cos(self.theta / 2)

        return self.r * (1 - cos(asin(self.x / (2 * self.r))))

    @property
    def x(self):
        if self._x is not None:
            return self._x

        if self._y2 is not None:
            return 2 * self.r * sin(2 * asin(sqrt(self.y2 / (2 * self.r))))

        return 2 * self.r * sin(self.theta / 2) if self._x is None else self._x

    @property
    def area(self):
        return ((self.r**2) / 2) * (self.theta - sin(self.theta))

    def point(self, phi):
        """
        Return a point on the segment, located at an angle phi within the segment.

        It is assumed that:

        * The segment is located at the top of the circle.
        * The angle phi is taken anti-clockwise from the RHS most extreme point of
            the segment.

        :param phi: the angle at which the point is calculated. Must be between 0
            and theta.
        """

        if not 0 <= phi <= self.theta:
            raise ValueError("Expected phi to be in the range of 0 <= phi <= theta")

        start_angle = (pi - self.theta) / 2
        angle = start_angle + phi

        return self.r * cos(angle), self.r * sin(angle)

    @property
    def circle(self):
        """
        Return a full circle that the segment is based on.
        """

        return Circle(r=self.r)

    def inverse_segment(self):
        """
        Return the segment that corresponds to the other side of the chord line.
        """

        theta = 2 * pi - self.theta

        return CircularSegment(r=self.r, theta=theta)


if __name__ == "__main__":
    c1 = CircularSegment(r=1.888, y2=0.1)
    print("Chord c1")
    print(c1.theta)
    print(c1.y2)
    print(c1.x)
    print()

    c2 = CircularSegment(r=1.888, theta=c1.theta)
    print("Chord c2")
    print(c2.y2)
    print()

    c3 = CircularSegment(r=1.888, theta=2 * pi)
    print("Chord c3")
    print(c3.theta)
    print(c3.y2)
    print(c3.x)
    print()

    c4 = CircularSegment(r=1.888, y2=2 * 1.888 - 0.1)
    print("Chord c4")
    print(c4.theta)
    print(c4.y2)
    print(c4.x)
    print()


def build_circle(
    *,
    centroid=Point | tuple[float, float],
    radius,
    no_points: int = 64,
    limit_angles: tuple[float, float] | None = None,
    use_radians: bool = True,
) -> list[tuple[float, float]]:
    """
    Build a list of points that approximate a circle or circular arc.

    :param centroid: The centroid of the circle.
    :param radius: The radius of the circle.
    :param no_points: The no. of points to include in the definition of the circle.
    :param limit_angles: Angles to limit the circular arc. Should be of the format
        (min, max).
        Angles to be taken CCW.
    :param use_radians: Use radians for angles?
    :return: A circle, or part thereof, as a list of lists defining the points:
        [[x1, y1], [x2, y2], ..., [xn, yn]]
    """

    full_circle = radians(360)

    if limit_angles is not None:
        min_angle = limit_angles[0]
        max_angle = limit_angles[1]

        if not use_radians:
            min_angle = radians(min_angle)
            max_angle = radians(max_angle)

    else:
        min_angle = 0
        max_angle = full_circle

    angle_range = np.linspace(start=min_angle, stop=max_angle, num=no_points)
    x_points_orig = np.full(no_points, radius)

    x_points = x_points_orig * np.cos(angle_range)  # - y_points * np.sin(angle_range)
    y_points = x_points_orig * np.sin(angle_range)  # + y_points * np.cos(angle_range)
    # can neglect the 2nd half of the formula because the y-points are just zeroes

    if isinstance(centroid, Point):
        centroid = (centroid.x, centroid.y)

    x_points = x_points + centroid[0]
    y_points = y_points + centroid[1]

    all_points = np.transpose(np.stack((x_points, y_points)))

    return [(p[0], p[1]) for p in all_points.tolist()]
