"""
Tests of the AS2304 module
"""

from math import isclose

import pytest

from utilityscripts.loads.as2304 import Tank

tank_50_30 = Tank(
    diameter=50.0,
    height=30.0,
    freeboard=0.0,
    w_shell=100,
    x_shell=15.0,
    w_roof=100,
    x_roof=30.0,
)
tank_30_30 = Tank(
    diameter=30.0,
    height=30.0,
    freeboard=0.0,
    w_shell=100,
    x_shell=15.0,
    w_roof=100,
    x_roof=30.0,
)
tank_10_20 = Tank(
    diameter=5.093 * 2,
    height=20.0,
    freeboard=0.0,
    w_shell=100,
    x_shell=15.0,
    w_roof=100,
    x_roof=30.0,
)  # based on test model in Strand7 with a radius of 32/pi/2


def test_tank():
    d = 30.0
    h = 30.0
    fb = 0.0
    w_s = 100.0
    x_s = 15.0
    w_r = 100.0
    x_r = 30.0
    gamma_l = 10.0

    tank = Tank(
        diameter=d,
        height=h,
        freeboard=fb,
        w_shell=w_s,
        x_shell=x_s,
        w_roof=w_r,
        x_roof=x_r,
        gamma_l=gamma_l,
    )

    assert tank.diameter == d
    assert tank.height == h
    assert tank.freeboard == fb
    assert tank.w_shell == w_s
    assert tank.x_shell == x_s
    assert tank.w_roof == w_r
    assert tank.x_roof == x_r
    assert tank.gamma_l == gamma_l
    assert tank.d == d
    assert tank.radius == d / 2


@pytest.mark.parametrize(
    "tank, expected",
    [
        (
            tank_50_30,
            0.620,
        ),
        (
            tank_30_30,
            0.782,
        ),
    ],
)
def test_s4_6_2_1_alpha_1(tank, expected):
    assert isclose(tank.alpha_1, expected, rel_tol=1e-2)


@pytest.mark.parametrize(
    "tank, expected",
    [
        (
            tank_50_30,
            0.374,
        ),
        (
            tank_30_30,
            0.230,
        ),
    ],
)
def test_s4_6_2_1_alpha_2(tank, expected):
    assert isclose(tank.alpha_2, expected, rel_tol=1e-2)


@pytest.mark.parametrize(
    "tank, expected",
    [
        (
            tank_50_30,
            0.375 * 30.0,
        ),
        (
            tank_30_30,
            (0.50 - 0.094) * 30.0,
        ),
    ],
)
def test_s4_6_2_1_x_1(tank, expected):
    assert isclose(tank.x_1, expected, rel_tol=1e-2)


@pytest.mark.parametrize(
    "tank, expected",
    [
        (
            tank_50_30,
            19.08,
        ),
        (
            tank_30_30,
            22.23,
        ),
    ],
)
def test_s4_6_2_1_x_2(tank, expected):
    assert isclose(tank.x_2, expected, rel_tol=1e-2)


@pytest.mark.parametrize(
    "tank, y, k_p, z, expected",
    [
        (tank_50_30, 0.0, 1.25, 0.12, 0.0),
        (tank_50_30, 3.0, 1.25, 0.12, 90.8),
        (tank_50_30, 15.0, 1.25, 0.12, 359),
        (tank_50_30, 27.0, 1.25, 0.12, 473),
        (tank_50_30, 30.0, 1.25, 0.12, 478),
        (tank_30_30, 0.0, 1.25, 0.12, 0.0),
        (tank_30_30, 3.0, 1.25, 0.12, 49.1),
        (tank_30_30, 15.0, 1.25, 0.12, 175),
        (tank_30_30, 27.0, 1.25, 0.12, 198),
        (tank_30_30, 30.0, 1.25, 0.12, 198),
    ],
)
def test_s4_6_3_p1(tank, y, k_p, z, expected):
    assert isclose(tank.s4_6_3_p_1(y=y, k_p=k_p, z=z), expected, rel_tol=1e-2)


@pytest.mark.parametrize(
    "tank, y, k_p, z, s, expected",
    [
        (tank_50_30, 0.0, 1.25, 0.12, 1.00, 36.8),
        (tank_50_30, 3.0, 1.25, 0.12, 1.00, 29.7),
        (tank_50_30, 15.0, 1.25, 0.12, 1.00, 13.4),
        (tank_50_30, 27.0, 1.25, 0.12, 1.00, 8.19),
        (tank_50_30, 30.0, 1.25, 0.12, 1.00, 8.00),
        (tank_30_30, 0.0, 1.25, 0.12, 1.00, 22.6),
        (tank_30_30, 3.0, 1.25, 0.12, 1.00, 15.7),
        (tank_30_30, 15.0, 1.25, 0.12, 1.00, 3.68),
        (tank_30_30, 27.0, 1.25, 0.12, 1.00, 1.22),
        (tank_30_30, 30.0, 1.25, 0.12, 1.00, 1.14),
    ],
)
def test_s4_6_3_p2(tank, y, k_p, z, s, expected):
    assert isclose(tank.s4_6_3_p_2(y=y, k_p=k_p, z=z, s=s), expected, rel_tol=1e-2)


@pytest.mark.parametrize(
    "tank, y, expected",
    [
        (tank_30_30, 0.0, 0.0),
        (tank_30_30, 5.0, 750.0),
        (tank_30_30, 15.0, 2250.0),
        (tank_30_30, 30.0, 4500.0),
        (tank_10_20, 0, 0),
        (tank_10_20, 20.0, 1016.56),  # calculated in Strand7
    ],
)
def test_p_hydrostatic(tank, y, expected):
    assert isclose(tank.p_hydrostatic(y=y), expected, rel_tol=1e-2)
