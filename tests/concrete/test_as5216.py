"""
Tests of the as5216 module.
"""

from math import isclose

import pytest

from utilityscripts.concrete.as5216 import phi_ms_shear, phi_ms_tension


@pytest.mark.parametrize(
    "f_yf, f_uf, expected",
    [
        (240, 400, 0.5),  # 4.6 bolt
        (660, 830, 0.66265),  # 8.8 bolt
        (750, 830, 1 / 1.4),  # case of a high f ratio.
    ],
)
def test_phi_m_s_tension(f_yf, f_uf, expected):
    assert isclose(phi_ms_tension(f_yf=f_yf, f_uf=f_uf), expected, rel_tol=1e-3)


@pytest.mark.parametrize(
    "f_yf, f_uf, expected",
    [
        (240, 400, 0.6),
        (660, 830, 2 / 3),  # case of f_uf > 800, 8.8 grade.
        (360, 400, 2 / 3),  # case of f_ratio > 0.8
    ],
)
def test_phi_m_s_shear(f_yf, f_uf, expected):
    assert isclose(phi_ms_shear(f_yf=f_yf, f_uf=f_uf), expected, rel_tol=1e-3)
