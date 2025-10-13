"""
Test functions in the design module.
"""

from math import isclose

import pytest

from utilityscripts.steel.design import (
    AS4100,
    CornerDetail,
    ISection,
    SteelMember,
    make_section,
)
from utilityscripts.steel.steel import steel_grades

g300 = steel_grades()["AS/NZS3679.1:300"]


@pytest.mark.parametrize(
    "designation, b_f, d, t_f, t_w, corner_detail, corner_size, n_r",
    [
        ("test1", 0.1, 0.15, 0.015, 0.01, None, None, 4),
        ("test2", 0.1, 0.15, 0.015, 0.01, CornerDetail.RADIUS, 0.010, 8),
        ("test3", 0.1, 0.15, 0.015, 0.01, CornerDetail.WELD, 0.005, 16),
    ],
)
def test_i_section(designation, b_f, d, t_f, t_w, corner_detail, corner_size, n_r):
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
        n_r=n_r,
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
    assert i_section.n_r == n_r


@pytest.mark.parametrize(
    "designation, grade, area_expected",
    [
        ("310UB40.4", "AS/NZS3679.1:300", 0.00521),
        ("310UB40.4", g300, 0.00521),
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

    new_section = _310ub40.add_steel(g300)

    assert _310ub40.steel is None
    assert new_section.steel == g300

    assert isclose(new_section.area_gross, 0.00521, rel_tol=1e-3)


def test_steel_member():
    """
    Test that steel member works.
    """

    member = SteelMember()

    assert member.section is None
    assert member.length is None
    assert member.restraints is None

    _310ub40 = make_section(designation="310UB40.4")

    new_member = member.add_section(_310ub40)

    assert member.section is None
    assert new_member.section == _310ub40

    length = 2.0

    new_member = member.add_length(length)

    assert member.length is None
    assert new_member.length == length


def test_as4100():
    """
    Basic test of the AS4100 class.
    """

    design = AS4100()

    assert design.member is None
    assert design.length is None

    length = 2.0
    member = SteelMember(section=None, length=length, restraints=None)

    new_design = design.add_member(member)

    assert design.member is None
    assert new_design.member is member


def test_as4100_tension():
    _310ub40 = make_section(designation="310UB40.4", grade=g300)
    length = 2.0

    member = SteelMember(section=_310ub40, length=length, restraints=None)

    design = AS4100(member=member)

    assert isclose(design.n_ty().value, 1666600, rel_tol=1e-3)
    assert isclose(design.phi_n_ty(phi_steel=0.9).value, 1499900, rel_tol=1e-3)
