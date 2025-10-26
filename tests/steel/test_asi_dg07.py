"""
Tests related to ASI DG07 - pinned baseplates
"""

from math import isclose

import pytest

from utilityscripts.steel.asi_dg07 import (
    chk04_v_cf,
    chk10_a_n_group,
    chk10_a_no,
    chk10_phi_n_c2,
)


@pytest.mark.parametrize(
    "n_star, a_c, f_c, mu, phi, phi_c, expected",
    [
        (1400, 350 * 350, 32, 0.4, 0.7, 0.7, 392),  # DG07 example 11.1
        (
            14000,
            350 * 350,
            32,
            0.4,
            0.7,
            0.7,
            472.2,
        ),  # DG07 example 11.1, increased compression so that 5.50 x a_c governs.
        (
            14000,
            350 * 350,
            25,
            0.4,
            0.7,
            0.7,
            428.8,
        ),  # increased compression, reduced f_c, so that concrete capacity governs.
        (120, 550 * 230, 25, 0.4, 0.7, 0.7, 33.6),  # DG07 example 11.2
    ],
)
def test_chk09(n_star, a_c, f_c, mu, phi, phi_c, expected):
    assert isclose(
        chk04_v_cf(n_star=n_star, a_c=a_c, f_c=f_c, mu=mu, phi=phi, phi_c=phi_c),
        expected,
        rel_tol=1e-2,
    )


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
