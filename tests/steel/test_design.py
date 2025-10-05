"""
Test functions in the design module.
"""

from math import isclose

from utilityscripts.steel.design import make_section
from utilityscripts.steel.steel import steel_grades


def test_make_section():
    """
    Preliminary test that make_section works. Use a 310UB40.4 section.
    """

    g300 = steel_grades()["AS/NZS3679.1:300"]
    _310ub40 = make_section(designation="310UB40.4", grade=g300)

    assert isclose(_310ub40.area_gross, 0.00521, rel_tol=1e-3)


def test_add_steel():
    _310ub40 = make_section(designation="310UB40.4")
    assert isclose(_310ub40.area_gross, 0.00521, rel_tol=1e-3)
    assert _310ub40.steel is None

    g300 = steel_grades()["AS/NZS3679.1:300"]

    new_section = _310ub40.add_steel(g300)

    assert _310ub40.steel is None
    assert new_section.steel == g300

    assert isclose(new_section.area_gross, 0.00521, rel_tol=1e-3)
