"""
Some tests for the concrete module.
"""

from math import isclose

from utilityscripts.concrete import reo_area

REL_TOL = 0.01


def test_bar_area_nos():

    assert isclose(reo_area("N20"), 314, rel_tol=REL_TOL)
    assert isclose(reo_area("2-N20"), 628, rel_tol=REL_TOL)
    assert isclose(reo_area("1-N36"), 1020, rel_tol=REL_TOL)
    assert isclose(reo_area("9-N32"), 7200, rel_tol=REL_TOL)


def test_bar_area_spacing():

    assert isclose(reo_area("N20-200"), 1570, rel_tol=REL_TOL)
    assert isclose(reo_area("N16-380"), 526, rel_tol=REL_TOL)
    assert isclose(reo_area("N36-160"), 6375, rel_tol=REL_TOL)


def test_bar_area_mesh():

    assert isclose(reo_area("SL82"), 227, rel_tol=REL_TOL)
    assert isclose(reo_area("SL82", main_direction=False), 227, rel_tol=REL_TOL)
    assert isclose(reo_area("RL1218"), 1112, rel_tol=REL_TOL)
    assert isclose(reo_area("RL1218", main_direction=False), 227, rel_tol=REL_TOL)
