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


def test_segment():

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


def test_segment_2():
    """
    Some tests based on models in CAD
    """

    test_chords = {
        "c1": {"r": 25.0, "theta": 2.36322, "y2": 15.5140887, "x": 46.26089},
        "c2": {"r": 50.0, "theta": 2.94126, "y2": 45.0, "x": 99.498744},
        "c3": {"r": 30.0, "theta": 1.24124, "y2": 5.594487, "x": 34.892459},
        "c4": {"r": 12.5, "theta": pi, "y2": 12.5, "x": 25.0},
    }

    tolerance = 0.00001

    for test, vals in test_chords.items():

        c = Circular_Segment(r=vals["r"], theta=vals["theta"])

        assert isclose(c.r, vals["r"], rel_tol=tolerance)
        assert isclose(c.theta, vals["theta"], rel_tol=tolerance)
        assert isclose(c.y2, vals["y2"], rel_tol=tolerance)
        assert isclose(c.x, vals["x"], rel_tol=tolerance)
