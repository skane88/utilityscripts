"""
Test functions in loads.py
"""

from math import isclose

import pytest

from utilityscripts.loads.as522x import (
    DriveClass,
    HoistClass,
    s5_2_a_h,
    s5_2_f_h,
    s6_1_2_1_beta_2,
    s6_1_2_1_hoist_class,
    s6_1_2_1_phi_2,
    s6_1_2_1_phi_2_min,
)


@pytest.mark.parametrize(
    "delta, expected",
    [
        (0.000, HoistClass.HC4),
        (0.100, HoistClass.HC4),
        (0.149, HoistClass.HC4),
        (0.150, HoistClass.HC3),
        (0.200, HoistClass.HC3),
        (0.299, HoistClass.HC3),
        (0.300, HoistClass.HC2),
        (0.500, HoistClass.HC2),
        (0.799, HoistClass.HC2),
        (0.800, HoistClass.HC1),
        (100.000, HoistClass.HC1),
    ],
)
def test_get_hoist_class(delta: float, expected: HoistClass):
    assert s6_1_2_1_hoist_class(delta=delta) == expected


@pytest.mark.parametrize(
    "hoist_class, expected",
    [
        (HoistClass.HC1, 0.17),
        (HoistClass.HC2, 0.34),
        (HoistClass.HC3, 0.51),
        (HoistClass.HC4, 0.68),
    ],
)
def test_beta_2(hoist_class: HoistClass, expected: float):
    assert isclose(s6_1_2_1_beta_2(hoist_class=hoist_class), expected)


@pytest.mark.parametrize(
    "hoist_class, drive_class, expected",
    [
        (HoistClass.HC1, DriveClass.HD1, 1.05),
        (HoistClass.HC1, DriveClass.HD2, 1.05),
        (HoistClass.HC1, DriveClass.HD3, 1.05),
        (HoistClass.HC1, DriveClass.HD4, 1.05),
        (HoistClass.HC1, DriveClass.HD5, 1.05),
        (HoistClass.HC2, DriveClass.HD1, 1.10),
        (HoistClass.HC2, DriveClass.HD2, 1.10),
        (HoistClass.HC2, DriveClass.HD3, 1.05),
        (HoistClass.HC2, DriveClass.HD4, 1.10),
        (HoistClass.HC2, DriveClass.HD5, 1.05),
        (HoistClass.HC3, DriveClass.HD1, 1.15),
        (HoistClass.HC3, DriveClass.HD2, 1.15),
        (HoistClass.HC3, DriveClass.HD3, 1.05),
        (HoistClass.HC3, DriveClass.HD4, 1.15),
        (HoistClass.HC3, DriveClass.HD5, 1.05),
        (HoistClass.HC4, DriveClass.HD1, 1.20),
        (HoistClass.HC4, DriveClass.HD2, 1.20),
        (HoistClass.HC4, DriveClass.HD3, 1.05),
        (HoistClass.HC4, DriveClass.HD4, 1.20),
        (HoistClass.HC4, DriveClass.HD5, 1.05),
    ],
)
def test_phi_2_min(hoist_class: HoistClass, drive_class: DriveClass, expected: float):
    assert isclose(
        s6_1_2_1_phi_2_min(hoist_class=hoist_class, drive_class=drive_class), expected
    )


@pytest.mark.parametrize(
    "m_h, expected",
    [
        (1000, 0.5),
        (2000, 1.0),
    ],
)
def test_as5222_s5_2_a_h(m_h, expected):
    assert isclose(s5_2_a_h(m_h=m_h), expected)


@pytest.mark.parametrize(
    "v_h, hoist_class, drive_class, expected",
    [(0.200, HoistClass.HC4, DriveClass.HD1, 1.336)],
)
def test_phi_2(v_h, hoist_class, drive_class, expected):
    assert isclose(
        s6_1_2_1_phi_2(v_h=v_h, hoist_class=hoist_class, drive_class=drive_class),
        expected,
        rel_tol=1e-3,
    )


@pytest.mark.parametrize(
    "q_z, a_h, c_h, expected",
    [
        (240, 0.5, 2.4, 288.0),  # corresonds to a 20m/s windspeed
        (15, 1, 2.0, 30.0),
    ],  # corresponds to a 5m/s windspeed
)
def test_as5222_s5_2_f_h(q_z, a_h, c_h, expected):
    assert isclose(s5_2_f_h(q_z=q_z, a_h=a_h, c_h=c_h), expected)
