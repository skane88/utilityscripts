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

    assert isclose(reo_area(bar_spec), area, rel_tol=REL_TOL)


@pytest.mark.parametrize(
    "bar_spec,area,main_direction",
    [
        ("SL82", 227, True),
        ("SL82", 227, False),
        ("RL1218", 1112, True),
        ("RL1218", 227, False),
    ],
)
def test_bar_area_mesh(bar_spec, area, main_direction):

    assert isclose(
        reo_area(bar_spec, main_direction=main_direction), area, rel_tol=REL_TOL
    )
