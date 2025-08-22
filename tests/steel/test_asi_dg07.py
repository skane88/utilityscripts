"""
Tests related to ASI DG07 - pinned baseplates
"""

from math import isclose

from utilityscripts.steel.asi_dg07 import chk10_a_n_group, chk10_a_no, chk10_phi_n_c2


def test_a_n_o():
    expected = chk10_a_no(h_ef=325.0)
    assert isclose(expected, 9 * 325 * 325, rel_tol=1e-4)


def test_a_n_group():
    expected = chk10_a_n_group(h_ef=325.0, s_p=230.0, s_g=230.0, t_c=500.0)
    assert isclose(expected, 1452025.0, rel_tol=1e-4)


def test_n_c2():
    """
    Test AIC Eloise slab (4 bolt group)
    """

    expected = chk10_phi_n_c2(
        n_b=4, h_ef=325.0, s_p=230.0, s_g=230.0, t_c=500.0, f_c=32.0
    )
    assert isclose(expected, 369.44, rel_tol=1e-4)
