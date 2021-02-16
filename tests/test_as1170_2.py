"""
Contains some tests for the AS1170.2 module.
"""

from pathlib import Path

import toml
import pytest

from as1170_2 import V_R

FILE_PATH = Path(__file__)
TEST_DATA_PATH = FILE_PATH.parent / Path("test_as1170_2.toml")
TEST_DATA = toml.load(TEST_DATA_PATH)


def build_V_R_pairs():

    test_data = TEST_DATA["regional_windspeeds"]

    pairs = []

    for region, v in test_data.items():

        for R, V_R in v.items():

            pairs.append([(region, R), V_R])

    return pairs


@pytest.mark.parametrize("input, expected", build_V_R_pairs())
def test_V_R_no_F_x(input, expected):
    """
    Basic test of the V_R method. Ignores F_x because that's how the data I have is formatted.
    """

    region = input[0]
    R = int(input[1])

    V_R_calc = round(
        V_R(wind_region=region, R=R, ignore_F_x=True)
    )  # round because the data from AS1170 is rounded

    assert V_R_calc == expected
