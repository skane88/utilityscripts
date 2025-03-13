from math import isclose

import pytest

from utilityscripts.concrete.ccaa_t48 import (
    LoadingType,
    LoadLocation,
    MaterialFactor,
    e_se,
    e_sl_from_cbr,
    e_ss_from_e_sl,
    f_e,
    f_h,
    f_s,
    k_1,
    k_2,
    k_3,
    k_4,
    t_4,
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


@pytest.mark.parametrize(
    "e_sl, b, expected",
    [
        (21, 0.4, 52.5),
        (35, 0.8, 43.75),
    ],
)
def test_e_ss_from_e_sl(e_sl, b, expected):
    assert isclose(e_ss_from_e_sl(e_sl=e_sl, b=b), expected, rel_tol=1e-3)


@pytest.mark.parametrize(
    "cbr, expected",
    [
        (2, 9.5),
        (10, 28),
        (20, 36),
        (40, 61),
        (60, 84),
    ],
)
def test_e_sl_from_cbr(cbr, expected):
    assert isclose(e_sl_from_cbr(cbr=cbr), expected, rel_tol=1e-2)


@pytest.mark.parametrize(
    "cbr, expected ",
    [
        (1, 10),
        (100, 27.5),
    ],
)
def test_s_sl_from_cbr_errors(cbr, expected):
    with pytest.raises(ValueError):
        assert isclose(e_sl_from_cbr(cbr=cbr), expected, rel_tol=1e-2)


@pytest.mark.parametrize(
    "load_location, expected",
    [
        (LoadLocation.INTERNAL, 1.2),
        (LoadLocation.EDGE, 1.05),
    ],
)
def test_k_3(load_location, expected):
    assert isclose(k_3(load_location=load_location), expected, rel_tol=1e-2)


@pytest.mark.parametrize(
    "f_c, expected",
    [
        (20, 1.03),
        (22.5, 1.05),
        (25, 1.07),
        (32, 1.11),
        (40, 1.16),
        (45, 1.18),
        (50, 1.20),
    ],
)
def test_k_4(f_c, expected):
    assert isclose(k_4(f_c=f_c), expected, rel_tol=1e-2)


@pytest.mark.parametrize("f_c, expected", [(10, 10), (65, 10)])
def test_k_4_errors(f_c, expected):
    with pytest.raises(ValueError):
        assert isclose(k_4(f_c=f_c), expected, rel_tol=1e-2)


@pytest.mark.parametrize(
    "e_ss, load_type, load_location, expected",
    [
        (25, LoadingType.WHEEL, LoadLocation.INTERNAL, 1.19),
        (100, LoadingType.WHEEL, LoadLocation.INTERNAL, 1.5375),
        (10.0, LoadingType.WHEEL, LoadLocation.EDGE, 1.00),
        (100, LoadingType.WHEEL, LoadLocation.EDGE, 1.68),
        (30.5, LoadingType.POINT, LoadLocation.INTERNAL, 1.26),
        (90, LoadingType.POINT, LoadLocation.INTERNAL, 1.65),
        (4.6, LoadingType.POINT, LoadLocation.EDGE, 0.68),
        (80, LoadingType.POINT, LoadLocation.EDGE, 1.464),
        (10.4, LoadingType.DISTRIBUTED, LoadLocation.INTERNAL, 0.83),
        (94.8, LoadingType.DISTRIBUTED, LoadLocation.INTERNAL, 3.06),
        (10.4, LoadingType.DISTRIBUTED, LoadLocation.EDGE, 0.83),
        (94.8, LoadingType.DISTRIBUTED, LoadLocation.EDGE, 3.06),
    ],
)
def test_f_e(e_ss, load_type, load_location, expected):
    assert isclose(
        f_e(e_ss=e_ss, load_type=load_type, load_location=load_location),
        expected,
        rel_tol=1e-2,
    )


@pytest.mark.parametrize(
    "e_ss, load_type, load_location, expected",
    [
        (2.5, LoadingType.WHEEL, LoadLocation.INTERNAL, 1.0),
        (160, LoadingType.WHEEL, LoadLocation.INTERNAL, 1.0),
        (2.0, LoadingType.WHEEL, LoadLocation.EDGE, 1.0),
        (160, LoadingType.WHEEL, LoadLocation.EDGE, 1.0),
        (2.5, LoadingType.POINT, LoadLocation.INTERNAL, 1.0),
        (110, LoadingType.POINT, LoadLocation.INTERNAL, 1.0),
        (1.0, LoadingType.POINT, LoadLocation.EDGE, 1.0),
        (110, LoadingType.POINT, LoadLocation.EDGE, 1.0),
        (2.5, LoadingType.DISTRIBUTED, LoadLocation.INTERNAL, 1.0),
        (160, LoadingType.DISTRIBUTED, LoadLocation.INTERNAL, 1.0),
        (2.5, LoadingType.DISTRIBUTED, LoadLocation.EDGE, 1.0),
        (160, LoadingType.DISTRIBUTED, LoadLocation.EDGE, 1.0),
    ],
)
def test_f_e_errors(e_ss, load_type, load_location, expected):
    with pytest.raises(ValueError):
        assert isclose(
            f_e(e_ss=e_ss, load_type=load_type, load_location=load_location),
            expected,
            rel_tol=1e-2,
        )


@pytest.mark.parametrize(
    "x, load_type, load_location, expected",
    [
        (1.22, LoadingType.WHEEL, LoadLocation.INTERNAL, 0.95),
        (2.85, LoadingType.WHEEL, LoadLocation.INTERNAL, 1.16),
        (1.20, LoadingType.WHEEL, LoadLocation.EDGE, 0.956),
        (2.90, LoadingType.WHEEL, LoadLocation.EDGE, 1.123),
        (1.27, LoadingType.POINT, LoadLocation.INTERNAL, 0.89),
        (2.90, LoadingType.POINT, LoadLocation.INTERNAL, 1.47),
        (1.27, LoadingType.POINT, LoadLocation.EDGE, 0.89),
        (2.90, LoadingType.POINT, LoadLocation.EDGE, 1.47),
        (1.07, LoadingType.DISTRIBUTED, LoadLocation.INTERNAL, 1.55),
        (4.58, LoadingType.DISTRIBUTED, LoadLocation.INTERNAL, 0.88),
        (1.07, LoadingType.DISTRIBUTED, LoadLocation.EDGE, 1.55),
        (4.58, LoadingType.DISTRIBUTED, LoadLocation.EDGE, 0.88),
    ],
)
def test_f_s(x, load_type, load_location, expected):
    assert isclose(
        f_s(x=x, load_type=load_type, load_location=load_location),
        expected,
        rel_tol=1e-2,
    )


@pytest.mark.parametrize(
    "x, load_type, load_location, expected",
    [
        (0.5, LoadingType.WHEEL, LoadLocation.INTERNAL, 1.0),
        (5.0, LoadingType.WHEEL, LoadLocation.INTERNAL, 1.0),
        (0.5, LoadingType.WHEEL, LoadLocation.EDGE, 1.0),
        (5.0, LoadingType.WHEEL, LoadLocation.EDGE, 1.0),
        (0.5, LoadingType.POINT, LoadLocation.INTERNAL, 1.0),
        (5.0, LoadingType.POINT, LoadLocation.INTERNAL, 1.0),
        (0.5, LoadingType.POINT, LoadLocation.EDGE, 1.0),
        (5.0, LoadingType.POINT, LoadLocation.EDGE, 1.0),
        (0.5, LoadingType.DISTRIBUTED, LoadLocation.INTERNAL, 1.0),
        (6.0, LoadingType.DISTRIBUTED, LoadLocation.INTERNAL, 1.0),
        (0.5, LoadingType.DISTRIBUTED, LoadLocation.EDGE, 1.0),
        (6.0, LoadingType.DISTRIBUTED, LoadLocation.EDGE, 1.0),
    ],
)
def test_f_s_errors(x, load_type, load_location, expected):
    with pytest.raises(ValueError):
        assert isclose(
            f_s(x=x, load_type=load_type, load_location=load_location),
            expected,
            rel_tol=1e-2,
        )


@pytest.mark.parametrize(
    "h, load_type, load_location, expected",
    [
        (1.0, LoadingType.WHEEL, LoadLocation.INTERNAL, 1.19),
        (15.5, LoadingType.WHEEL, LoadLocation.INTERNAL, 0.98),
        (0.7, LoadingType.WHEEL, LoadLocation.EDGE, 1.25),
        (15.2, LoadingType.WHEEL, LoadLocation.EDGE, 0.99),
        (0.7, LoadingType.POINT, LoadLocation.INTERNAL, 1.58),
        (11.0, LoadingType.POINT, LoadLocation.INTERNAL, 0.97),
        (0.7, LoadingType.POINT, LoadLocation.EDGE, 1.58),
        (11.0, LoadingType.POINT, LoadLocation.EDGE, 0.97),
        (3.0, LoadingType.DISTRIBUTED, LoadLocation.INTERNAL, 1.1),
        (15.5, LoadingType.DISTRIBUTED, LoadLocation.INTERNAL, 0.91),
        (3.0, LoadingType.DISTRIBUTED, LoadLocation.EDGE, 1.1),
        (15.5, LoadingType.DISTRIBUTED, LoadLocation.EDGE, 0.91),
    ],
)
def test_f_h(h, load_type, load_location, expected):
    assert isclose(
        f_h(h=h, load_type=load_type, load_location=load_location),
        expected,
        rel_tol=1e-2,
    )


@pytest.mark.parametrize(
    "h, load_type, load_location, expected",
    [
        (0.25, LoadingType.WHEEL, LoadLocation.INTERNAL, 1.0),
        (17.0, LoadingType.WHEEL, LoadLocation.INTERNAL, 1.0),
        (0.25, LoadingType.WHEEL, LoadLocation.EDGE, 1.0),
        (17.0, LoadingType.WHEEL, LoadLocation.EDGE, 1.0),
        (0.25, LoadingType.POINT, LoadLocation.INTERNAL, 1.0),
        (17.0, LoadingType.POINT, LoadLocation.INTERNAL, 1.0),
        (0.25, LoadingType.POINT, LoadLocation.EDGE, 1.0),
        (17.0, LoadingType.POINT, LoadLocation.EDGE, 1.0),
        (0.25, LoadingType.DISTRIBUTED, LoadLocation.INTERNAL, 1.0),
        (17.0, LoadingType.DISTRIBUTED, LoadLocation.INTERNAL, 1.0),
        (0.25, LoadingType.DISTRIBUTED, LoadLocation.EDGE, 1.0),
        (17.0, LoadingType.DISTRIBUTED, LoadLocation.EDGE, 1.0),
    ],
)
def test_f_h_errors(h, load_type, load_location, expected):
    with pytest.raises(ValueError):
        assert isclose(
            f_h(h=h, load_type=load_type, load_location=load_location),
            expected,
            rel_tol=1e-2,
        )


@pytest.mark.parametrize(
    "f_4, expected",
    [
        (40, 570),
        (70, 340),
        (97, 160),
    ],
)
def test_t_4(f_4, expected):
    assert isclose(t_4(f_4=f_4), expected, rel_tol=1e-2)


@pytest.mark.parametrize(
    "f_4, expected",
    [
        (30, 400),
        (110, 100),
    ],
)
def test_t_4_errors(f_4, expected):
    with pytest.raises(ValueError):
        assert isclose(t_4(f_4=f_4), expected, rel_tol=1e-2)
