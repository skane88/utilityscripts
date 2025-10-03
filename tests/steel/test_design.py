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
