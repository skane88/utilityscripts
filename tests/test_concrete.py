"""
Some tests for the concrete module.
"""

from math import isclose

import pytest

from utilityscripts.concrete import reo_area

REL_TOL = 0.01


@pytest.mark.parametrize(
    "bar_spec,area",
    [
        ("N20", 314),
        ("2-N20", 628),
        ("1-N36", 1020),
        ("9-N32", 7200),
        ("L12", 113),
        ("R12", 113),
        ("N20-200", 1570),
        ("N16-380", 526),
        ("N36-160", 6375),
    ],
)
def test_bar_area(bar_spec, area):
    assert isclose(reo_area(bar_spec)["total_area"], area, rel_tol=REL_TOL)


@pytest.mark.parametrize(
    "bar_spec,area,main_direction",
    [
        ("SL82", 227, True),
        ("SL82", 227, False),
        ("RL1218", 1112, True),
        ("RL1218", 227, False),
        ("F1218", 1227, True),
        ("F1218", 251, False),
    ],
)
def test_bar_area_mesh(bar_spec, area, main_direction):
    assert isclose(
        reo_area(bar_spec, main_direction=main_direction)["total_area"],
        area,
        rel_tol=REL_TOL,
    )


@pytest.mark.parametrize(
    "bar_spec,width,main_direction,expected",
    [
        (
            "SL82",
            1200,
            True,
            {
                "single_bar_area": 45.4,
                "total_area": 272,
                "area_unit_width": 227,
                "no_bars": 6.0,
                "width": 1200,
            },
        ),
        (
            "SL82",
            1200,
            False,
            {
                "single_bar_area": 45.4,
                "total_area": 272,
                "area_unit_width": 227,
                "no_bars": 6.0,
                "width": 1200,
            },
        ),
    ],
)
def test_bar_area_full(bar_spec, width, main_direction, expected):
    return_vals = reo_area(
        bar_spec=bar_spec, main_direction=main_direction, width=width
    )

    assert isclose(
        return_vals["single_bar_area"], expected["single_bar_area"], rel_tol=REL_TOL
    )
    assert isclose(
        return_vals["total_area"],
        expected["total_area"],
        rel_tol=REL_TOL,
    )
    assert isclose(
        return_vals["area_unit_width"], expected["area_unit_width"], rel_tol=REL_TOL
    )
    assert isclose(return_vals["no_bars"], expected["no_bars"], rel_tol=REL_TOL)
    assert isclose(return_vals["width"], expected["width"], rel_tol=REL_TOL)
