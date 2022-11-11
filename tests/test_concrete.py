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


def test_bar_area_mesh():

    assert isclose(reo_area("SL82"), 227, rel_tol=REL_TOL)
    assert isclose(reo_area("SL82", main_direction=False), 227, rel_tol=REL_TOL)
    assert isclose(reo_area("RL1218"), 1112, rel_tol=REL_TOL)
    assert isclose(reo_area("RL1218", main_direction=False), 227, rel_tol=REL_TOL)
    assert isclose(reo_area("SL82", width=1200), 272, rel_tol=REL_TOL)
