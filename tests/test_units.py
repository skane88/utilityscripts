"""
File to test functions in the units module.
"""

from math import isclose

import pytest

from utilityscripts.units import inch, kg, m, m_ceil_units, m_floor_units, m_round_units


@pytest.mark.parametrize(
    "x, base, expected",
    [
        (0.1275 * m, 0.005 * m, 0.130 * m),
        (0.126 * m, 0.005 * m, 0.125 * m),
        (0.129 * m, 0.005 * m, 0.130 * m),
        (-0.1275 * m, 0.005 * m, -0.130 * m),
        (0.1275 * m, 0.005, 0.130 * m),
        (0.126 * m, 0.005, 0.125 * m),
        (0.129 * m, 0.005, 0.130 * m),
        (-0.1275 * m, 0.005, -0.130 * m),
        (5 * inch, 0.005 * m, (0.125 * m).to(inch)),
        (5 * inch, 2, 4.0 * inch),
        (4.9 * inch, 2, 4.0 * inch),
        (5.1 * inch, 2, 6.0 * inch),
    ],
)
def test_m_round_units(x, base, expected):
    assert isclose(m_round_units(x, base).magnitude, expected.magnitude)
    assert m_round_units(x, base).units == expected.units


@pytest.mark.parametrize(
    "x, base, expected, error",
    [(5, 2 * m, None, ValueError), (5 * kg, 2 * m, None, ValueError)],
)
def test_m_round_units_error(x, base, expected, error):
    with pytest.raises(error):
        assert m_round_units(x, base) == expected


@pytest.mark.parametrize(
    "x, base, expected",
    [
        (0.1275 * m, 0.005 * m, 0.125 * m),
        (0.126 * m, 0.005 * m, 0.125 * m),
        (0.129 * m, 0.005 * m, 0.125 * m),
        (-0.1275 * m, 0.005 * m, -0.130 * m),
        (0.1275 * m, 0.005, 0.125 * m),
        (0.126 * m, 0.005, 0.125 * m),
        (0.129 * m, 0.005, 0.125 * m),
        (-0.1275 * m, 0.005, -0.130 * m),
        (5 * inch, 0.005 * m, (0.125 * m).to(inch)),
        (5 * inch, 2, 4.0 * inch),
        (4.9 * inch, 2, 4.0 * inch),
        (5.1 * inch, 2, 4.0 * inch),
    ],
)
def test_m_floor_units(x, base, expected):
    assert isclose(m_floor_units(x, base).magnitude, expected.magnitude)
    assert m_floor_units(x, base).units == expected.units


@pytest.mark.parametrize(
    "x, base, expected",
    [
        (0.1275 * m, 0.005 * m, 0.130 * m),
        (0.126 * m, 0.005 * m, 0.130 * m),
        (0.129 * m, 0.005 * m, 0.130 * m),
        (-0.1275 * m, 0.005 * m, -0.125 * m),
        (0.1275 * m, 0.005, 0.130 * m),
        (0.126 * m, 0.005, 0.130 * m),
        (0.129 * m, 0.005, 0.130 * m),
        (-0.1275 * m, 0.005, -0.125 * m),
        (5 * inch, 0.005 * m, (0.130 * m).to(inch)),
        (5 * inch, 2, 6.0 * inch),
        (4.9 * inch, 2, 6.0 * inch),
        (5.1 * inch, 2, 6.0 * inch),
    ],
)
def test_m_ceil_units(x, base, expected):
    assert isclose(m_ceil_units(x, base).magnitude, expected.magnitude)
    assert m_ceil_units(x, base).units == expected.units
