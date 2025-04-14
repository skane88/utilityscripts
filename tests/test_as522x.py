"""
Test functions in loads.py
"""

from math import isclose

import pytest

from utilityscripts.loads.as522x import (
    DriveClass,
    HoistClass,
    get_beta_2,
    get_hoist_class,
    get_phi_2_min,
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
    assert get_hoist_class(delta=delta) == expected


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
    assert isclose(get_beta_2(hoist_class=hoist_class), expected)


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
        get_phi_2_min(hoist_class=hoist_class, drive_class=drive_class), expected
    )
