"""
Module to contain tests for code in steel.py
"""

from math import isclose, radians

import pytest

from utilityscripts.steel import Lug


def create_lug():
    """
    Create a basic lug for use in later tests.
    """

    return Lug(t=0.016, b=0.170, h=0.060, r=0.05, dia_hole=0.034)


def create_eccentric():
    """
    Create a basic lug for use in later tests.
    """

    return Lug(t=0.016, b=0.170, h=0.060, r=0.05, dia_hole=0.034, e_hole=-0.035)


@pytest.mark.parametrize(
    "lug, load, out_of_plane_load, expected",
    [
        (
            create_lug(),
            1.0,
            0.0,
            (-0.433013, 0.75, -0.5, -0.025981, 0.066651, 0.021160),
        ),
        (
            create_eccentric(),
            1.0,
            0.0,
            (-0.433013, 0.75, -0.5, 0.000269, 0.066651, 0.038660),
        ),
        (
            create_lug(),
            1.0,
            0.04,
            (-0.433013, 0.75, -0.46, -0.025981, 0.062519, 0.020160),
        ),
        (
            create_eccentric(),
            1.0,
            0.04,
            (-0.433013, 0.75, -0.46, 0.000269, 0.062519, 0.036260),
        ),
    ],
)
def test_lug_load_resolve(lug, load, out_of_plane_load, expected):
    """
    Test the method for resolving loads on the lug.
    """

    results = lug.resolve_load(
        load=load,
        out_of_plane_load=out_of_plane_load,
        angle_in_plane=radians(30),
        angle_out_of_plane=radians(30),
        hole_offset=0.05,
        out_of_plane_offset=0.020,
    )

    zipped = zip(results, expected)

    for z in zipped:
        assert isclose(z[0], z[1], rel_tol=0.001)
