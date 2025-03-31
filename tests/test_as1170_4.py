"""
Test the earthquake module.
"""

from math import isclose

import pytest

from utilityscripts.earthquake.as1170_4 import SoilType, spectral_shape_factor


@pytest.mark.parametrize(
    "soil, period, expected",
    [
        (SoilType.Ae, 0.0, 0.80),
        (SoilType.Ae, 0.1, 2.35),
        (SoilType.Ae, 0.2, 2.35),
        (SoilType.Ae, 0.3, 2.35),
        (SoilType.Ae, 0.4, 1.76),
        (SoilType.Ae, 0.5, 1.41),
        (SoilType.Ae, 1.0, 0.70),
        (SoilType.Ae, 1.5, 0.47),
        (SoilType.Ae, 2.0, 0.26),
        (SoilType.Ae, 3.0, 0.12),
        (SoilType.Ae, 4.0, 0.066),
        (SoilType.Ae, 5.0, 0.042),
        (SoilType.Be, 0.0, 1.00),
        (SoilType.Be, 0.1, 2.94),
        (SoilType.Be, 0.2, 2.94),
        (SoilType.Be, 0.3, 2.94),
        (SoilType.Be, 0.4, 2.20),
        (SoilType.Be, 0.5, 1.76),
        (SoilType.Be, 1.0, 0.88),
        (SoilType.Be, 1.5, 0.59),
        (SoilType.Be, 2.0, 0.33),
        (SoilType.Be, 3.0, 0.15),
        (SoilType.Be, 4.0, 0.083),
        (SoilType.Be, 5.0, 0.053),
        (SoilType.Ce, 0.0, 1.30),
        (SoilType.Ce, 0.1, 3.68),
        (SoilType.Ce, 0.2, 3.68),
        (SoilType.Ce, 0.3, 3.68),
        (SoilType.Ce, 0.4, 3.12),
        (SoilType.Ce, 0.5, 2.50),
        (SoilType.Ce, 1.0, 1.25),
        (SoilType.Ce, 1.5, 0.83),
        (SoilType.Ce, 2.0, 0.47),
        (SoilType.Ce, 3.0, 0.21),
        (SoilType.Ce, 4.0, 0.12),
        (SoilType.Ce, 5.0, 0.075),
        (SoilType.De, 0.0, 1.10),
        (SoilType.De, 0.1, 3.68),
        (SoilType.De, 0.2, 3.68),
        (SoilType.De, 0.3, 3.68),
        (SoilType.De, 0.4, 3.68),
        (SoilType.De, 0.5, 3.68),
        (SoilType.De, 1.0, 1.98),
        (SoilType.De, 1.5, 1.32),
        (SoilType.De, 2.0, 0.74),
        (SoilType.De, 3.0, 0.33),
        (SoilType.De, 4.0, 0.19),
        (SoilType.De, 5.0, 0.12),
        (SoilType.Ee, 0.0, 1.10),
        (SoilType.Ee, 0.1, 3.68),
        (SoilType.Ee, 0.2, 3.68),
        (SoilType.Ee, 0.3, 3.68),
        (SoilType.Ee, 0.4, 3.68),
        (SoilType.Ee, 0.5, 3.68),
        (SoilType.Ee, 1.0, 3.08),
        (SoilType.Ee, 1.5, 2.05),
        (SoilType.Ee, 2.0, 1.14),
        (SoilType.Ee, 3.0, 0.51),
        (SoilType.Ee, 4.0, 0.29),
        (SoilType.Ee, 5.0, 0.18),
    ],
)
def test_spectral_shape_factor(soil: SoilType, period: float, expected: float):
    assert isclose(
        spectral_shape_factor(soil_type=soil, period=period), expected, abs_tol=2e-2
    )
