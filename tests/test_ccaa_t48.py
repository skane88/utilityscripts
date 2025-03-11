from math import isclose

import pytest

from utilityscripts.concrete.ccaa_t48 import (
    LoadingType,
    MaterialFactor,
    e_se,
    k_1,
    k_2,
    w_fi,
)


@pytest.mark.parametrize(
    "loading_type, material_factor, expected",
    [
        (LoadingType.WHEEL, MaterialFactor.CONSERVATIVE, 0.85),
        (LoadingType.WHEEL, MaterialFactor.UNCONSERVATIVE, 0.95),
        (LoadingType.WHEEL, MaterialFactor.MIDRANGE, 0.9),
        (LoadingType.POINT, MaterialFactor.CONSERVATIVE, 0.75),
        (LoadingType.POINT, MaterialFactor.UNCONSERVATIVE, 0.85),
        (LoadingType.POINT, MaterialFactor.MIDRANGE, 0.8),
        (LoadingType.DISTRIBUTED, MaterialFactor.CONSERVATIVE, 0.75),
        (LoadingType.DISTRIBUTED, MaterialFactor.UNCONSERVATIVE, 0.85),
        (LoadingType.DISTRIBUTED, MaterialFactor.MIDRANGE, 0.8),
    ],
)
def test_k_1(loading_type, material_factor, expected):
    assert isclose(
        k_1(loading_type=loading_type, material_factor=material_factor),
        expected,
        rel_tol=1e-6,
    )


@pytest.mark.parametrize(
    "no_cycles, load_type, expected",
    [
        (1e99, LoadingType.WHEEL, 0.50),
        (4e5, LoadingType.WHEEL, 0.51),
        (3e5, LoadingType.WHEEL, 0.52),
        (2e5, LoadingType.WHEEL, 0.54),
        (1e5, LoadingType.WHEEL, 0.56),
        (5e4, LoadingType.WHEEL, 0.59),
        (3e4, LoadingType.WHEEL, 0.61),
        (1e4, LoadingType.WHEEL, 0.65),
        (2e3, LoadingType.WHEEL, 0.70),
        (1e3, LoadingType.WHEEL, 0.73),
        (50, LoadingType.WHEEL, 0.84),
        (1, LoadingType.WHEEL, 1.00),
        (1e99, LoadingType.DISTRIBUTED, 0.75),
        (1e5, LoadingType.DISTRIBUTED, 0.75),
        (1, LoadingType.DISTRIBUTED, 0.75),
        (1e99, LoadingType.WHEEL, 0.50),
        (4e5, LoadingType.POINT, 0.51),
        (3e5, LoadingType.POINT, 0.52),
        (2e5, LoadingType.POINT, 0.54),
        (1e5, LoadingType.POINT, 0.56),
        (5e4, LoadingType.POINT, 0.59),
        (3e4, LoadingType.POINT, 0.61),
        (1e4, LoadingType.POINT, 0.65),
        (2e3, LoadingType.POINT, 0.70),
        (1e3, LoadingType.POINT, 0.73),
        (50, LoadingType.POINT, 0.84),
        (1, LoadingType.POINT, 1.00),
    ],
)
def test_k2(no_cycles, load_type, expected):
    assert isclose(
        k_2(no_cycles=no_cycles, load_type=load_type), expected, rel_tol=1e-2
    )


@pytest.mark.parametrize(
    "depth, normalising_length, loading_type, expected",
    [
        (1.0, 1.0, LoadingType.WHEEL, 0.675),
        (1.0, 1.0, LoadingType.POINT, 0.76),
        (1.0, 1.0, LoadingType.DISTRIBUTED, 0.78),
        (0.0, 1.0, LoadingType.WHEEL, 1.00),
        (0.0, 1.0, LoadingType.POINT, 1.00),
        (0.0, 1.0, LoadingType.DISTRIBUTED, 1.00),
        (12.0, 1.0, LoadingType.WHEEL, 0.05),
        (12.0, 1.0, LoadingType.POINT, 0.05),
        (12.0, 1.0, LoadingType.DISTRIBUTED, 0.14),
    ],
)
def test_w_fi(depth, normalising_length, loading_type, expected):
    assert isclose(
        w_fi(
            depth=depth,
            normalising_length=normalising_length,
            loading_type=loading_type,
        ),
        expected,
        rel_tol=1e-2,
    )


def test_e_se():
    h_layers = [1.5, 2.5, 2.0, 3.0]
    e_layers = [20, 42, 37.4, 59.5]
    wheel_spacing = 1.8
    loading = LoadingType.WHEEL
    expected = 31.8

    actual = e_se(
        h_layers=h_layers,
        e_layers=e_layers,
        normalising_length=wheel_spacing,
        loading_type=loading,
    )

    assert isclose(actual, expected, rel_tol=1e-3)
