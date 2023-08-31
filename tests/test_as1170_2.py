"""
Contains some tests for the AS1170.2 module.
"""

from math import isclose
from pathlib import Path

import pytest
import toml
from as1170_2 import V_R, M_d

FILE_PATH = Path(__file__)
TEST_DATA_PATH = FILE_PATH.parent / Path("test_as1170_2.toml")
TEST_DATA = toml.load(TEST_DATA_PATH)


def build_V_R_pairs(V_R_data):
    test_data = TEST_DATA[V_R_data]

    pairs = []

    for region, v in test_data.items():
        pairs.extend([(region, r), v_r] for r, v_r in v.items())
    return pairs


@pytest.mark.parametrize(
    "input_vals, expected", build_V_R_pairs("regional_windspeeds_no_F_x")
)
def test_V_R_no_F_x(input_vals, expected):
    """
    Basic test of the V_R method.
    Ignores F_x because that's how the data I have is formatted.
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
    Basic test of the V_R method.
    Ignores F_x because that's how the data I have is formatted.
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
        for direction, m_d_val in v.items():
            if direction.lower() != "any":
                direction = float(direction)

            m_d_val = tuple(m_d_val)

            pairs.append([(region, direction), m_d_val])

            if isinstance(direction, str):
                continue

            pairs.extend(
                (
                    [(region, direction + 360), m_d_val],
                    [(region, direction - 360), m_d_val],
                )
            )

    return pairs


@pytest.mark.parametrize("input_vals", build_direction_pairs())
def test_m_d(input_vals):
    wind_region = input_vals[0][0]
    direction = input_vals[0][1]
    expected = input_vals[1]

    m_d_calc = M_d(wind_region=wind_region, direction=direction)

    assert isclose(m_d_calc[0], expected[0])
    assert isclose(m_d_calc[1], expected[1])


def test_M_zcat_basic():
    assert False


def test_M_zcat_ave():
    assert False


def test_M_s():
    assert False
