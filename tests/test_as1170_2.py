"""
Contains some tests for the AS1170.2 module.
"""

from pathlib import Path
from math import isclose

import toml
import pytest

from as1170_2 import V_R, M_d

FILE_PATH = Path(__file__)
TEST_DATA_PATH = FILE_PATH.parent / Path("test_as1170_2.toml")
TEST_DATA = toml.load(TEST_DATA_PATH)


def build_V_R_pairs(V_R_data):

    test_data = TEST_DATA[V_R_data]

    pairs = []

    for region, v in test_data.items():

        for R, V_R in v.items():

            pairs.append([(region, R), V_R])

    return pairs


@pytest.mark.parametrize(
    "input_vals, expected", build_V_R_pairs("regional_windspeeds_no_F_x")
)
def test_V_R_no_F_x(input_vals, expected):
    """
    Basic test of the V_R method. Ignores F_x because that's how the data I have is formatted.
    """

    region = input_vals[0]
    R = int(input_vals[1])

    V_R_calc = round(
        V_R(wind_region=region, R=R, ignore_F_x=True)
    )  # round because the data from AS1170 is rounded

    assert V_R_calc == expected


@pytest.mark.parametrize("input_vals, expected", build_V_R_pairs("regional_windspeeds"))
def test_V_R(input_vals, expected):
    """
    Basic test of the V_R method. Ignores F_x because that's how the data I have is formatted.
    """

    region = input_vals[0]
    R = int(input_vals[1])

    V_R_calc = V_R(wind_region=region, R=R, ignore_F_x=False)

    V_R_calc = round(V_R_calc)  # round because the data from AS1170 is rounded

    assert V_R_calc == expected


def build_direction_pairs():

    test_data = TEST_DATA["wind_direction_factor"]

    pairs = []

    for region, v in test_data.items():

        for direction, M_d in v.items():

            if direction.lower() != "any":
                direction = float(direction)

            M_d = tuple(M_d)

            pairs.append([(region, direction), M_d])

            if isinstance(direction, str):
                continue

            pairs.append([(region, direction + 360), M_d])
            pairs.append([(region, direction - 360), M_d])

    return pairs


@pytest.mark.parametrize("input_vals, expected", build_direction_pairs())
def test_M_d(input_vals, expected):

    wind_region = input_vals[0]
    direction = input_vals[1]

    M_d_calc = M_d(wind_region=wind_region, direction=direction)

    assert isclose(M_d_calc[0], expected[0])
    assert isclose(M_d_calc[1], expected[1])


def test_M_zcat_basic():

    assert False


def test_M_zcat_ave():

    assert False


def test_M_s():

    assert False
