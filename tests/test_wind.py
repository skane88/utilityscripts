"""
Contains some tests for the AS1170.2 module.
"""

from math import isclose
from pathlib import Path

import pytest
import toml
from wind import V_R, M_d

FILE_PATH = Path(__file__)
TEST_DATA_PATH_2011 = FILE_PATH.parent / Path("test_as1170_2_2011.toml")
TEST_DATA_2011 = toml.load(TEST_DATA_PATH_2011)

TEST_DATA_PATH_2021 = FILE_PATH.parent / Path("test_as1170_2_2021.toml")
TEST_DATA_2021 = toml.load(TEST_DATA_PATH_2021)


def build_V_R_pairs_2011(V_R_data):
    test_data = TEST_DATA_2011[V_R_data]

    pairs = []

    for region, v in test_data.items():
        pairs.extend([(region, r), v_r] for r, v_r in v.items())
    return pairs


@pytest.mark.parametrize(
    "input_vals, expected", build_V_R_pairs_2011("regional_windspeeds_no_F_x")
)
def test_V_R_no_F_x_2011(input_vals, expected):
    """
    Basic test of the V_R method.
    Ignores F_x because that's how the data I have is formatted.
    """

    region = input_vals[0]
    R = int(input_vals[1])

    V_R_calc = round(
        V_R(wind_region=region, R=R, ignore_M_c=True, version="2011")
    )  # round because the data from AS1170 is rounded

    assert V_R_calc == expected


@pytest.mark.parametrize(
    "input_vals, expected", build_V_R_pairs_2011("regional_windspeeds")
)
def test_V_R_2011(input_vals, expected):
    """
    Basic test of the V_R method.
    Ignores F_x because that's how the data I have is formatted.
    """

    region = input_vals[0]
    R = int(input_vals[1])

    V_R_calc = V_R(wind_region=region, R=R, ignore_M_c=False, version="2011")

    V_R_calc = round(V_R_calc)  # round because the data from AS1170 is rounded

    assert V_R_calc == expected


def build_V_R_pairs_2021(V_R_data):
    test_data = TEST_DATA_2021[V_R_data]

    pairs = []

    for region, v in test_data.items():
        pairs.extend([(region, r), v_r] for r, v_r in v.items())
    return pairs


@pytest.mark.parametrize(
    "input_vals, expected", build_V_R_pairs_2021("regional_windspeeds")
)
def test_V_R_2021(input_vals, expected):
    """
    Basic test of the V_R method.
    Ignores F_x because that's how the data I have is formatted.
    """

    region = input_vals[0]
    R = int(input_vals[1])

    V_R_calc = V_R(wind_region=region, R=R, ignore_M_c=False, version="2021")

    V_R_calc = round(V_R_calc)  # round because the data from AS1170 is rounded

    print(f"Input:{input_vals}")
    print(f"Expected: {expected}")
    print(f"Calculated: {V_R_calc}")

    assert V_R_calc == expected


def build_direction_pairs_2011():
    test_data = TEST_DATA_2011["wind_direction_factor"]

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


@pytest.mark.parametrize("input_vals", build_direction_pairs_2011())
def test_m_d_2011(input_vals):
    wind_region = input_vals[0][0]
    direction = input_vals[0][1]
    expected = input_vals[1]

    m_d_calc = M_d(wind_region=wind_region, direction=direction, version="2011")

    assert isclose(m_d_calc[0], expected[0])
    assert isclose(m_d_calc[1], expected[1])


def build_direction_pairs_2021():
    test_data = TEST_DATA_2021["wind_direction_factor"]

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


@pytest.mark.parametrize("input_vals", build_direction_pairs_2021())
def test_m_d_2021(input_vals):
    wind_region = input_vals[0][0]
    direction = input_vals[0][1]
    expected = input_vals[1]

    m_d_calc = M_d(wind_region=wind_region, direction=direction, version="2021")

    assert isclose(m_d_calc[0], expected[0])
    assert isclose(m_d_calc[1], expected[1])


def test_M_zcat_basic():
    assert False


def test_M_zcat_ave():
    assert False


def test_M_s():
    assert False
