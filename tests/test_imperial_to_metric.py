"""
Test the imperial_to_metric methods.
"""

from math import isclose

import pytest

from utilityscripts.imperial_to_metric import feet_inches_to_m


@pytest.mark.parametrize(
    "example,expected",
    [
        (r"23'", 7.0104),
        (r'12"', 0.3048),
        (r'11.0"', 0.2794),
        (r'11 1/2"', 0.2921),
        (r"1'" + r'6"', 0.4572),
        (r"12'" + r'9.5"', 3.8989),
        (r"1'" + r'6 1/2"', 0.4699),
        ("23'6 1/2\"", 7.1755),
    ],
)
def test_convert_to_metric(example, expected):

    assert isclose(feet_inches_to_m(example), expected)


@pytest.mark.parametrize(
    "example,expected",
    [
        (r"23' 6 1/2\"", 7.1755),
        (r'23\' 6 1/2"', 7.1755),
        (r'23 6 1/2"', 7.1755),
        (r"23\' 6 1/2", 7.1755),
    ],
)
def test_convert_to_metric_errors(example, expected):
    """
    Tests for errors expected to be raised by the function.
    """

    with pytest.raises(ValueError):
        assert isclose(feet_inches_to_m(example), expected)
