"""
Tests of the as5216 module.
"""

from math import isclose

import pytest

from utilityscripts.concrete.as5216 import (
    Concrete,
    cracked,
    phi_m_p,
    phi_m_sp,
    phi_mc,
    phi_ms_ca,
    phi_ms_flex,
    phi_ms_l,
    phi_ms_l_x,
    phi_ms_re,
    phi_ms_shear,
    phi_ms_tension,
)


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


def test_phi_ms_ca():
    assert isclose(phi_ms_ca(), 1 / 1.8, rel_tol=1e-3)


def test_phi_ms_l():
    assert isclose(phi_ms_l(), 1 / 1.8, rel_tol=1e-3)


@pytest.mark.parametrize("phi_inst, expected", [(1.0, 1 / 1.8), (0.5, 0.5 / 1.8)])
def test_phi_ms_l_x(phi_inst, expected):
    assert isclose(phi_ms_l_x(phi_inst=phi_inst), expected, rel_tol=1e-3)


def test_phi_ms_flex():
    assert isclose(phi_ms_flex(), 1 / 1.15, rel_tol=1e-3)


def test_phi_ms_re():
    assert isclose(phi_ms_re(), 0.8, rel_tol=1e-3)


@pytest.mark.parametrize("phi_inst, expected", [(1.0, 1 / 1.5), (0.5, 0.5 / 1.5)])
def test_phi_mc(phi_inst, expected):
    assert isclose(phi_mc(phi_inst), expected, rel_tol=1e-3)


@pytest.mark.parametrize("phi_inst, expected", [(1.0, 1 / 1.5), (0.5, 0.5 / 1.5)])
def test_phi_m_sp(phi_inst, expected):
    assert isclose(phi_m_sp(phi_inst), expected, rel_tol=1e-3)


@pytest.mark.parametrize("phi_inst, expected", [(1.0, 1 / 1.5), (0.5, 0.5 / 1.5)])
def test_phi_m_p(phi_inst, expected):
    assert isclose(phi_m_p(phi_inst), expected, rel_tol=1e-3)


@pytest.mark.parametrize(
    "sigma_l, sigma_r, f_ct, expected",
    [(5.0, 5.0, 0.0, Concrete.CRACKED), (0.5, 1.5, 3.0, Concrete.UNCRACKED)],
)
def test_cracked(sigma_l, sigma_r, f_ct, expected):
    assert cracked(sigma_l=sigma_l, sigma_r=sigma_r, f_ct=f_ct) == expected
