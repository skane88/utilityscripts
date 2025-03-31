"""
Test the earthquake module.
"""

from math import isclose

import pytest

from utilityscripts.earthquake.as1170_4 import SoilClass, cd_t, spectral_shape_factor


@pytest.mark.parametrize(
    "soil, period, expected",
    [
        (SoilClass.Ae, 0.0, 0.80),
        (SoilClass.Ae, 0.1, 2.35),
        (SoilClass.Ae, 0.2, 2.35),
        (SoilClass.Ae, 0.3, 2.35),
        (SoilClass.Ae, 0.4, 1.76),
        (SoilClass.Ae, 0.5, 1.41),
        (SoilClass.Ae, 1.0, 0.70),
        (SoilClass.Ae, 1.5, 0.47),
        (SoilClass.Ae, 2.0, 0.26),
        (SoilClass.Ae, 3.0, 0.12),
        (SoilClass.Ae, 4.0, 0.066),
        (SoilClass.Ae, 5.0, 0.042),
        (SoilClass.Be, 0.0, 1.00),
        (SoilClass.Be, 0.1, 2.94),
        (SoilClass.Be, 0.2, 2.94),
        (SoilClass.Be, 0.3, 2.94),
        (SoilClass.Be, 0.4, 2.20),
        (SoilClass.Be, 0.5, 1.76),
        (SoilClass.Be, 1.0, 0.88),
        (SoilClass.Be, 1.5, 0.59),
        (SoilClass.Be, 2.0, 0.33),
        (SoilClass.Be, 3.0, 0.15),
        (SoilClass.Be, 4.0, 0.083),
        (SoilClass.Be, 5.0, 0.053),
        (SoilClass.Ce, 0.0, 1.30),
        (SoilClass.Ce, 0.1, 3.68),
        (SoilClass.Ce, 0.2, 3.68),
        (SoilClass.Ce, 0.3, 3.68),
        (SoilClass.Ce, 0.4, 3.12),
        (SoilClass.Ce, 0.5, 2.50),
        (SoilClass.Ce, 1.0, 1.25),
        (SoilClass.Ce, 1.5, 0.83),
        (SoilClass.Ce, 2.0, 0.47),
        (SoilClass.Ce, 3.0, 0.21),
        (SoilClass.Ce, 4.0, 0.12),
        (SoilClass.Ce, 5.0, 0.075),
        (SoilClass.De, 0.0, 1.10),
        (SoilClass.De, 0.1, 3.68),
        (SoilClass.De, 0.2, 3.68),
        (SoilClass.De, 0.3, 3.68),
        (SoilClass.De, 0.4, 3.68),
        (SoilClass.De, 0.5, 3.68),
        (SoilClass.De, 1.0, 1.98),
        (SoilClass.De, 1.5, 1.32),
        (SoilClass.De, 2.0, 0.74),
        (SoilClass.De, 3.0, 0.33),
        (SoilClass.De, 4.0, 0.19),
        (SoilClass.De, 5.0, 0.12),
        (SoilClass.Ee, 0.0, 1.10),
        (SoilClass.Ee, 0.1, 3.68),
        (SoilClass.Ee, 0.2, 3.68),
        (SoilClass.Ee, 0.3, 3.68),
        (SoilClass.Ee, 0.4, 3.68),
        (SoilClass.Ee, 0.5, 3.68),
        (SoilClass.Ee, 1.0, 3.08),
        (SoilClass.Ee, 1.5, 2.05),
        (SoilClass.Ee, 2.0, 1.14),
        (SoilClass.Ee, 3.0, 0.51),
        (SoilClass.Ee, 4.0, 0.29),
        (SoilClass.Ee, 5.0, 0.18),
    ],
)
def test_spectral_shape_factor(soil: SoilClass, period: float, expected: float):
    assert isclose(
        spectral_shape_factor(soil_class=soil, period=period), expected, abs_tol=2e-2
    )


@pytest.mark.parametrize(
    "soil, period, k_p, z, s_p, mu, expected",
    [
        (SoilClass.Ae, 0.1, 1.0, 0.08, 0.77, 2.00, 0.072),
        (SoilClass.Be, 0.1, 1.0, 0.08, 0.77, 2.00, 0.091),
        (SoilClass.Ce, 0.1, 1.0, 0.08, 0.77, 2.00, 0.113),
        (SoilClass.De, 0.1, 1.0, 0.08, 0.77, 2.00, 0.113),
        (SoilClass.Ee, 0.1, 1.0, 0.08, 0.77, 2.00, 0.113),
    ],
)
def test_cd_t(
    soil: SoilClass,
    period: float,
    k_p: float,
    z: float,
    s_p: float,
    mu: float,
    expected: float,
):
    assert isclose(
        cd_t(soil_class=soil, period=period, k_p=k_p, z=z, s_p=s_p, mu=mu),
        expected,
        abs_tol=1e-2,
    )
