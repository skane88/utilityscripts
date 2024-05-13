"""
File to test the math utilities file
"""

import pytest

from utilityscripts.math_utils import m_floor, m_round


@pytest.mark.parametrize(
    "x, base,expected",
    [
        (100, 1, 100),
        (-100, 1, -100),
        (12.34, 0.5, 12.5),
        (-12.34, 0.5, -12.5),
        (12.34, -0.5, 12.5),
        (-12.34, -0.5, -12.5),
        (1.30, 0.05, 1.30),
        (0.1 * 3 * 10, 1, 3),
        (4.6, 0.2, 4.6),
        (1.39, 0.05, 1.40),
        (4.3, 0.25, 4.25),
    ],
)
def test_m_round(x, base, expected):
    assert m_round(x, base) == expected


@pytest.mark.parametrize(
    "x, base, expected",
    [
        (100, 1, 100),
        (-100, 1, -100),
        (12.34, 0.5, 12.0),
        (-12.34, 0.5, -12.5),
        (12.34, -0.5, 12.0),
        (-12.34, -0.5, -12.5),
        (1.30, 0.05, 1.30),
        (0.1 * 3 * 10, 1, 3),
        (4.6, 0.2, 4.6),
        (1.39, 0.05, 1.35),
        (4.3, 0.25, 4.25),
    ],
)
def test_m_floor(x, base, expected):
    assert m_floor(x, base) == expected
