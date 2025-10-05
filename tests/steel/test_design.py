"""
Test functions in the design module.
"""

from math import isclose

import pytest

from utilityscripts.steel.design import ISection, make_section, CornerDetail
from utilityscripts.steel.steel import steel_grades


@pytest.mark.parametrize(
    "designation, b_f, d, t_f, t_w, corner_detail, corner_size",
    [
        ("test1", 0.1, 0.15, 0.015, 0.01, None, None),
        ("test2", 0.1, 0.15, 0.015, 0.01, CornerDetail.RADIUS, 0.010),
        ("test3", 0.1, 0.15, 0.015, 0.01, CornerDetail.WELD, 0.005),
    ],
)
def test_i_section(designation, b_f, d, t_f, t_w, corner_detail, corner_size):
    """
    Basic test of the ISection class.
    """

    i_section = ISection(
        section_name=designation,
        steel=None,
        b_f=b_f,
        d=d,
        t_f=t_f,
        t_w=t_w,
        corner_detail=corner_detail,
        corner_size=corner_size,
    )

    if not isinstance(b_f, tuple):
        b_f = (b_f, b_f)

    if not isinstance(t_f, tuple):
        t_f = (t_f, t_f)

    assert i_section.section_name == designation
    assert i_section.steel is None
    assert i_section.b_f == b_f
    assert i_section.d == d
    assert i_section.t_f == t_f
    assert i_section.t_w == t_w
    assert i_section.b_ft == b_f[0]
    assert i_section.b_fb == b_f[1]
    assert i_section.t_ft == t_f[0]
    assert i_section.t_fb == t_f[1]
    assert i_section.corner_detail == corner_detail
    assert (
        i_section.corner_size == corner_size
        if corner_size is not None
        else i_section.corner_size == 0.000
    )


@pytest.mark.parametrize(
    "designation, grade, area_expected",
    [
        ("310UB40.4", "AS/NZS3679.1:300", 0.00521),
        ("310UB40.4", steel_grades()["AS/NZS3679.1:300"], 0.00521),
        ("1000WB215", None, 27.40e-03),
    ],
)
def test_make_section(designation, grade, area_expected):
    """
    Preliminary test that make_section works.
    """

    steel_section = make_section(designation=designation, grade=grade)

    assert isclose(steel_section.area_gross, area_expected, rel_tol=5e-3)


def test_add_steel():
    _310ub40 = make_section(designation="310UB40.4")
    assert isclose(_310ub40.area_gross, 0.00521, rel_tol=1e-3)
    assert _310ub40.steel is None

    g300 = steel_grades()["AS/NZS3679.1:300"]

    new_section = _310ub40.add_steel(g300)

    assert _310ub40.steel is None
    assert new_section.steel == g300

    assert isclose(new_section.area_gross, 0.00521, rel_tol=1e-3)
