"""
Test the imperial_to_metric methods.
"""

from math import isclose

import pytest

from utilityscripts.imperial_to_metric import feet_inches_to_m


@pytest.mark.parametrize(
    "example,expected", [(r"23'", 7.0104),
    (r'12"', 0.3048),
    (r'11.0"', 0.3048),
    (r'11 1/2"', 0.2921),
    (r"12'" + r'9.5"', 3.8989)]
)
def test_convert_to_metric(example, expected):

    assert isclose(feet_inches_to_m(example), expected)
