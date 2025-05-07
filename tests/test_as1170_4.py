"""
Test the earthquake module.
"""

from math import isclose

import numpy as np
import pytest

from utilityscripts.earthquake.as1170_4 import (
    SoilClass,
    cd_t,
    k_p_data,
    k_p_z,
    k_p_z_min,
    spectral_shape_factor,
)


@pytest.mark.parametrize(
    "p, expected",
    [
        (2500, 1.8),
        (2250, 1.75),
        (2000, 1.7),
        (1500, 1.5),
        (1000, 1.3),
        (800, 1.25),
        (500, 1.0),
        (250, 0.75),
        (200, 0.70),
        (100, 0.50),
        (50, 0.50),
        (np.asarray([2500, 800, 100]), np.asarray([1.8, 1.25, 0.5])),
    ],
)
def test_k_p(p: float | np.ndarray, expected: float | np.ndarray):
    if isinstance(p, float):
        assert isclose(k_p_data(p=p), expected, abs_tol=1e-2)

    if isinstance(p, np.ndarray):
        assert np.allclose(k_p_data(p=p), expected, atol=1e-2)


@pytest.mark.parametrize("p", [10000, 10000.0, np.asarray([10000, 500])])
def test_k_p_raises_error(p):
    with pytest.raises(ValueError):
        k_p_data(p=p)


@pytest.mark.parametrize(
    "p, expected",
    [
        (500, 0.08),
        (1000, 0.10),
        (1500, 0.12),
        (2000, 0.14),
        (2500, 0.15),
        (np.asarray([500, 2500]), np.asarray([0.08, 0.15])),
    ],
)
def test_k_p_z_min(p, expected):
    if isinstance(p, float):
        assert isclose(k_p_z_min(p=p), expected, abs_tol=1e-2)

    if isinstance(p, np.ndarray):
        assert np.allclose(k_p_z_min(p=p), expected, atol=1e-2)


@pytest.mark.parametrize("p", [10000, 10000.0, np.asarray([10000])])
def test_k_p_z_min_raises_error(p):
    with pytest.raises(ValueError):
        k_p_z_min(p=p)


@pytest.mark.parametrize(
    "p, z, min_kpz, expected",
    [
        (500, 0.08, True, 0.08),
        (500, 0.10, True, 0.10),
        (500, 0.08, False, 0.08),
        (500, 0.10, False, 0.10),
        (2500, 0.08, True, 0.15),
        (2500, 0.10, True, 0.18),
        (2500, 0.08, False, 0.144),
        (2500, 0.10, False, 0.18),
    ],
)
def test_k_p_z(p, z, min_kpz, expected):
    assert isclose(k_p_z(p=p, z=z, min_kpz=min_kpz), expected, abs_tol=1e-2)


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
    "soil, period, min_period, expected",
    [
        (SoilClass.Ae, 0.0, False, 0.80),
        (SoilClass.Ae, 0.0, True, 2.35),
        (SoilClass.Be, 0.0, False, 1.00),
        (SoilClass.Be, 0.0, True, 2.94),
        (SoilClass.Ce, 0.0, False, 1.30),
        (SoilClass.Ce, 0.0, True, 3.68),
        (SoilClass.De, 0.0, False, 1.10),
        (SoilClass.De, 0.0, True, 3.68),
        (SoilClass.Ee, 0.0, False, 1.10),
        (SoilClass.Ee, 0.0, True, 3.68),
    ],
)
def test_spectral_shape_factor_min(soil, period, min_period, expected):
    assert isclose(
        spectral_shape_factor(soil_class=soil, period=period, min_period=min_period),
        expected,
        abs_tol=2e-2,
    )


@pytest.mark.parametrize(
    "soil, period, k_p_z, s_p, mu, expected",
    [
        (SoilClass.Ae, 0.1, 0.08, 0.77, 2.00, 0.072),
        (SoilClass.Be, 0.1, 0.08, 0.77, 2.00, 0.091),
        (SoilClass.Ce, 0.1, 0.08, 0.77, 2.00, 0.113),
        (SoilClass.De, 0.1, 0.08, 0.77, 2.00, 0.113),
        (SoilClass.Ee, 0.1, 0.08, 0.77, 2.00, 0.113),
    ],
)
def test_cd_t(
    soil: SoilClass,
    period: float,
    k_p_z: float,
    s_p: float,
    mu: float,
    expected: float,
):
    assert isclose(
        cd_t(soil_class=soil, period=period, k_p_z=k_p_z, s_p=s_p, mu=mu),
        expected,
        abs_tol=1e-2,
    )
