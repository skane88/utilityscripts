"""
Contains some tests for the geometry module.
"""

from math import isclose

from geometry import Chord


def test_chord():

    radius = 1.75

    c1 = Chord(r=radius, y2=0.5)

    c2 = Chord(r=radius, x=c1.x)
    c3 = Chord(r=radius, theta=c1.theta)

    assert isclose(c1.area, 0.84309861652350)  # value calculated in Excel

    assert isclose(c1.r, c2.r)
    assert isclose(c1.r, c3.r)
    assert isclose(c1.y2, c2.y2)
    assert isclose(c1.y2, c3.y2)
    assert isclose(c1.x, c2.x)
    assert isclose(c1.x, c3.x)
    assert isclose(c1.area, c2.area)
    assert isclose(c1.area, c3.area)