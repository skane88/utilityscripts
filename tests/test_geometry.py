"""
Contains some tests for the geometry module.
"""

from math import isclose, pi

from geometry import Circle, Circular_Segment


def test_circle():

    c1 = Circle(r=1)

    assert isclose(c1.d, 2)
    assert isclose(c1.area, pi)
    assert isclose(c1.circumference, pi * 2)

    c2 = Circle(d=2)

    assert isclose(c2.r, 1)
    assert isclose(c1.area, pi)
    assert isclose(c1.circumference, pi * 2)


def test_chord():

    radius = 1.75

    c1 = Circular_Segment(r=radius, y2=0.5)

    c2 = Circular_Segment(r=radius, x=c1.x)
    c3 = Circular_Segment(r=radius, theta=c1.theta)

    assert isclose(c1.area, 0.84309861652350)  # value calculated in Excel

    assert isclose(c1.r, c2.r)
    assert isclose(c1.r, c3.r)
    assert isclose(c1.y2, c2.y2)
    assert isclose(c1.y2, c3.y2)
    assert isclose(c1.x, c2.x)
    assert isclose(c1.x, c3.x)
    assert isclose(c1.area, c2.area)
    assert isclose(c1.area, c3.area)
