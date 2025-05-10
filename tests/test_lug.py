"""
File with tests for lifting_lug.py
"""

from functools import lru_cache
from math import isclose, radians

import pytest

from utilityscripts.lifting_lug.lifting_lug import Lug, LugLoad
from utilityscripts.steel.steel import steel_grades


@lru_cache
def create_steel():
    """
    Create a steel grade for use in later tests.
    """

    return steel_grades()["AS/NZS3678:250"]


def create_lug():
    """
    Create a basic lug for use in later tests.
    """

    return Lug(
        thickness=0.016,
        b_base=0.170,
        h_hole=0.060,
        radius=0.05,
        dia_hole=0.034,
        material=create_steel(),
    )


def create_eccentric_lug():
    """
    Create an eccentric lug for use in later tests.
    """

    return Lug(
        thickness=0.016,
        b_base=0.170,
        h_hole=0.060,
        radius=0.05,
        dia_hole=0.034,
        e_hole=-0.035,
        material=create_steel(),
    )


def test_lugload():
    swl = 1.0
    dia_pin = 0.020
    out_of_plane_allowance = 0.04
    shackle_offset = 0.05
    out_of_plane_offset = 0.020
    in_plane_angle_limits = (-radians(30), radians(30))
    out_of_plane_angle_limits = (-radians(30), radians(30))

    ll = LugLoad(
        swl=swl,
        dia_pin=dia_pin,
        shackle_offset=shackle_offset,
        out_of_plane_offset=out_of_plane_offset,
        in_plane_angle_limits=in_plane_angle_limits,
        out_of_plane_angle_limits=out_of_plane_angle_limits,
        out_of_plane_allowance=out_of_plane_allowance,
        use_radians=True,
    )

    assert ll.swl == swl
    assert ll.dia_pin == dia_pin
    assert ll.shackle_offset == shackle_offset
    assert ll.out_of_plane_offset == out_of_plane_offset
    assert ll.in_plane_angle_limits == in_plane_angle_limits
    assert ll.out_of_plane_angle_limits == out_of_plane_angle_limits
    assert ll.out_of_plane_allowance == out_of_plane_allowance
    assert ll.min_in_plane_angle == in_plane_angle_limits[0]
    assert ll.max_in_plane_angle == in_plane_angle_limits[1]
    assert ll.min_out_of_plane_angle == out_of_plane_angle_limits[0]
    assert ll.max_out_of_plane_angle == out_of_plane_angle_limits[1]


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
            create_eccentric_lug(),
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
            create_eccentric_lug(),
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

    results = lug.resolve_load_about_base(
        load=load,
        out_of_plane_allowance=out_of_plane_load,
        angle_in_plane=radians(30),
        angle_out_of_plane=radians(30),
        hole_offset=0.05,
        out_of_plane_offset=0.020,
    )

    zipped = zip(results, expected, strict=True)

    for z in zipped:
        assert isclose(z[0], z[1], rel_tol=0.001)
