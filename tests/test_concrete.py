"""
Some tests for the concrete module.
"""

from math import isclose

from utilityscripts.concrete import reo_area


def test_bar_area_nos():

    assert isclose(reo_area("N20"), 314, rel_tol=0.001)
    assert isclose(reo_area("2-N20"), 628, rel_tol=0.001)


def test_bar_area_spacing():

    assert isclose(reo_area("N20-200"), 1570, rel_tol=0.001)


def test_bar_area_mesh():

    assert isclose(reo_area("SL82"), 227, rel_tol=0.001)
