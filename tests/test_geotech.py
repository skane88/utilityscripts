"""
Contains tests for the geotech module.
"""

from math import isclose, radians

from utilityscripts.geotech import q_ult


def test_q_ult():
    """
    Basic test of the q_ult method against example 8.1 in Craig's Soil Mechanics
    """

    B = 2.25
    D = 1.5

    c = 0
    gamma = 18
    phi = radians(38)

    N_c = 1
    N_gamma = 67
    N_q = 49

    s_gamma = 0.8

    q = gamma * D

    expected = 2408
    actual = q_ult(
        c=c, N_c=N_c, gamma=gamma, q=q, N_q=N_q, B=B, N_gamma=N_gamma, s_gamma=s_gamma
    )

    assert isclose(expected, actual, rel_tol=0.001)
