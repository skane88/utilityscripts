from math import isclose

import pytest

from utilityscripts.concrete.ccaa_t48 import (
    CCAA_T48,
    Load,
    LoadingType,
    LoadLocation,
    MaterialFactor,
    Slab,
    Soil,
    SoilProfile,
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
    k_s_from_cbr,
    t_3,
    t_4,
    t_12,
    w_fi,
)


def test_soil():
    e_ss = 44
    e_sl = 35
    soil = Soil(e_sl=e_sl, e_ss=e_ss)

    assert soil
    assert soil.e_sl == e_sl
    assert soil.e_ss == e_ss
    assert soil.soil_name is None

    soil = Soil(e_sl=e_sl, e_ss=e_ss, soil_name="Sand")

    assert soil
    assert soil.e_sl == e_sl
    assert soil.e_ss == e_ss
    assert soil.soil_name == "Sand"


def test_soil_profile():
    e_ss = 44
    e_sl = 35

    soil = Soil(e_sl=e_sl, e_ss=e_ss, soil_name="Sand")

    soil_profile = SoilProfile(h_layers=[2.0], soils=[soil])

    assert (
        soil_profile.e_sl(normalising_length=1.0, loading_type=LoadingType.POINT)
        == e_sl
    )
    assert (
        soil_profile.e_ss(normalising_length=1.0, loading_type=LoadingType.POINT)
        == e_ss
    )
    assert soil_profile.h_layers == [2.0]
    assert soil_profile.soils == [soil]


def test_soil_profile_2():
    """
    Test SoilProfile against example D1 in CCAA T48
    """

    h_layers = [1.5, 2.5, 2.0, 3.0]
    layer_names = ["Fill", "Sand", "Sandy Clay", "Very Stiff Clay"]
    e_sl_layers = [20, 42, 37.4, 59.5]
    b_layers = [1.0, 0.8, 0.7, 0.6]
    e_ss_layers = [
        e_ss_from_e_sl(e_sl=e_sl, b=b)
        for e_sl, b in zip(e_sl_layers, b_layers, strict=True)
    ]

    soils = [
        Soil(e_sl=e_sl, e_ss=e_ss, soil_name=soil_name)
        for e_sl, e_ss, soil_name in zip(
            e_sl_layers, e_ss_layers, layer_names, strict=True
        )
    ]

    soil_profile = SoilProfile(h_layers=h_layers, soils=soils)

    wheel_spacing = 1.8
    loading = LoadingType.WHEEL
    expected = 31.8

    assert isclose(
        soil_profile.e_sl(normalising_length=wheel_spacing, loading_type=loading),
        expected,
        rel_tol=1e-3,
    )


def test_soil_profile_3():
    """
    Test SoilProfile against example D3 in CCAA T48
    """

    h_layers = [2.0, 5.0]
    layer_names = ["Fill", "Clay"]
    e_sl_layers = [30.0, 18.0]
    b_layers = [0.4, 0.6]
    e_ss_layers = [
        e_ss_from_e_sl(e_sl=e_sl, b=b)
        for e_sl, b in zip(e_sl_layers, b_layers, strict=True)
    ]

    soils = [
        Soil(e_sl=e_sl, e_ss=e_ss, soil_name=soil_name)
        for e_sl, e_ss, soil_name in zip(
            e_sl_layers, e_ss_layers, layer_names, strict=True
        )
    ]

    soil_profile = SoilProfile(h_layers=h_layers, soils=soils)

    load_width = 4.0
    loading = LoadingType.DISTRIBUTED
    expected = 20.8

    assert isclose(
        soil_profile.e_sl(normalising_length=load_width, loading_type=loading),
        expected,
        rel_tol=1e-3,
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


def test_e_se_1():
    """
    Test e_se against example D1 in CCAA T48
    """

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


def test_e_se_2():
    """
    Test e_se against example D3 in CCAA T48
    """

    h_layers = [2.0, 5.0]
    e_layers = [30.0, 18.0]
    load_width = 4.0
    loading = LoadingType.DISTRIBUTED
    expected = 20.8

    actual = e_se(
        h_layers=h_layers,
        e_layers=e_layers,
        normalising_length=load_width,
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
    "f_12, p, load_location, expected",
    [
        (0.5, 50, LoadLocation.INTERNAL, 350),
        (4.0, 50, LoadLocation.INTERNAL, 98),
        (2.8, 800, LoadLocation.INTERNAL, 600),
        (5.0, 800, LoadLocation.INTERNAL, 420),
        (0.0, 0.0, LoadLocation.INTERNAL, 0.0),
        (2.5, 200, LoadLocation.INTERNAL, 295),
        (0.8, 40, LoadLocation.INTERNAL, 215),
        (0.5, 50, LoadLocation.EDGE, 500),
        (5.0, 50, LoadLocation.EDGE, 107),
        (3.9, 600, LoadLocation.EDGE, 590),
        (5.0, 600, LoadLocation.EDGE, 510),
        (0.0, 0.0, LoadLocation.EDGE, 0.0),
        (1.0, 48, LoadLocation.EDGE, 325),
        (2.8, 200, LoadLocation.EDGE, 400),
    ],
)
def test_t_12(f_12, p, load_location, expected):
    assert isclose(
        t_12(f_12=f_12, p=p, load_location=load_location), expected, rel_tol=1e-2
    )


@pytest.mark.parametrize(
    "f_3, load_location, expected",
    [
        (20, LoadLocation.INTERNAL, 500),
        (120, LoadLocation.INTERNAL, 105),
        (40, LoadLocation.EDGE, 490),
        (120, LoadLocation.EDGE, 170),
    ],
)
def test_t_3(f_3, load_location, expected):
    assert isclose(t_3(f_3=f_3, load_location=load_location), expected, rel_tol=1e-2)


@pytest.mark.parametrize(
    "f_3, load_location, expected",
    [
        (10, LoadLocation.INTERNAL, 400),
        (135, LoadLocation.INTERNAL, 400),
        (25, LoadLocation.EDGE, 400),
        (135, LoadLocation.EDGE, 400),
    ],
)
def test_t_3_errors(f_3, load_location, expected):
    with pytest.raises(ValueError):
        assert isclose(
            t_3(f_3=f_3, load_location=load_location), expected, rel_tol=1e-2
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


@pytest.mark.parametrize(
    "cbr, expected",
    [
        (1.8, 21.0),
        (10, 53.0),
        (20, 68.0),
        (30, 83.0),
        (40, 109.0),
        (50, 135.0),
        (60, 156.0),
        (70, 174.0),
        (80, 189.0),
        (90, 201.0),
        (100, 215.0),
    ],
)
def test_k_s_from_cbr(cbr, expected):
    assert isclose(k_s_from_cbr(cbr=cbr), expected, rel_tol=1e-2)


def test_load():
    load = Load(
        load_type=LoadingType.WHEEL,
        magnitude=100.0,
        normalising_length=1.0,
        no_cycles=1e5,
    )
    assert load.load_type == LoadingType.WHEEL
    assert load.magnitude == 100.0  # noqa: PLR2004
    assert load.normalising_length == 1.0
    assert load.no_cycles == 1e5  # noqa: PLR2004


def test_slab():
    f_tf = 0.7 * (40**0.5)
    thickness = 0.200
    slab = Slab(f_tf=f_tf, thickness=thickness)

    assert slab.f_tf == f_tf
    assert slab.thickness == thickness


def test_ccaa():
    slab = Slab(f_tf=0.7 * (40**0.5), thickness=0.200)

    check_slab = CCAA_T48(
        slab=slab,
        material_factor=MaterialFactor.CONSERVATIVE,
        loads=None,
        soil_profile=None,
    )

    assert check_slab.slab == slab
    assert check_slab.material_factor == MaterialFactor.CONSERVATIVE
    assert check_slab.loads == {}
    assert check_slab.soil_profile is None


def test_ccaa_add_loads():
    soil_profile = SoilProfile(
        h_layers=[1.0, 1.0],
        soils=[
            Soil(e_sl=100000.0, e_ss=100000.0, soil_name="soil1"),
            Soil(e_sl=100000.0, e_ss=100000.0, soil_name="soil2"),
        ],
    )
    slab = Slab(f_tf=0.7 * (40**0.5), thickness=0.200)

    check_slab = CCAA_T48(slab=slab, soil_profile=soil_profile)

    assert check_slab.soil_profile == soil_profile
    assert check_slab.loads == {}

    check_slab_2 = check_slab.add_load(
        load_id="load_1",
        load_type=LoadingType.WHEEL,
        magnitude=100.0,
        normalising_length=1.0,
        no_cycles=1e5,
    )
    assert check_slab.loads == {}
    assert check_slab_2.loads == {
        "load_1": Load(
            load_type=LoadingType.WHEEL,
            magnitude=100.0,
            normalising_length=1.0,
            no_cycles=1e5,
        )
    }

    check_slab_3 = check_slab.add_loads(
        loads={
            "load_1": Load(
                load_type=LoadingType.WHEEL,
                magnitude=100.0,
                normalising_length=1.0,
                no_cycles=2e5,
            )
        }
    )
    assert check_slab.loads == {}
    assert check_slab_3.loads == {
        "load_1": Load(
            load_type=LoadingType.WHEEL,
            magnitude=100.0,
            normalising_length=1.0,
            no_cycles=2e5,
        )
    }


def test_ccaa_add_load_error():
    soil_profile = SoilProfile(
        h_layers=[1.0, 1.0],
        soils=[
            Soil(e_sl=100000.0, e_ss=100000.0, soil_name="soil1"),
            Soil(e_sl=100000.0, e_ss=100000.0, soil_name="soil2"),
        ],
    )
    slab = Slab(f_tf=0.7 * (40**0.5), thickness=0.200)

    check_slab = CCAA_T48(
        slab=slab,
        soil_profile=soil_profile,
    )
    check_slab = check_slab.add_load(
        load_id="load_1",
        load_type=LoadingType.WHEEL,
        magnitude=100.0,
        normalising_length=1.0,
        no_cycles=1e5,
    )

    with pytest.raises(ValueError):
        check_slab.add_load(
            load_id="load_1",
            load_type=LoadingType.WHEEL,
            magnitude=100.0,
            normalising_length=1.0,
            no_cycles=1e5,
        )


def test_ccaa_add_loads_error():
    soil_profile = SoilProfile(
        h_layers=[1.0, 1.0],
        soils=[
            Soil(e_sl=100000.0, e_ss=100000.0, soil_name="soil1"),
            Soil(e_sl=100000.0, e_ss=100000.0, soil_name="soil2"),
        ],
    )
    slab = Slab(f_tf=0.7 * (40**0.5), thickness=0.200)

    check_slab = CCAA_T48(
        slab=slab,
        soil_profile=soil_profile,
    )

    check_slab = check_slab.add_loads(
        loads={
            "load_1": Load(
                load_type=LoadingType.WHEEL,
                magnitude=100.0,
                normalising_length=1.0,
                no_cycles=1e5,
            ),
        }
    )

    with pytest.raises(ValueError):
        check_slab.add_loads(
            loads={
                "load_1": Load(
                    load_type=LoadingType.WHEEL,
                    magnitude=100.0,
                    normalising_length=1.0,
                    no_cycles=1e5,
                ),
            }
        )


def test_ccaa_app_d1():
    forklift = Load(
        load_type=LoadingType.WHEEL,
        magnitude=100,
        normalising_length=1.8,
        no_cycles=20 * 40 * 5 * 52,
    )

    h_layers = [1.5, 2.5, 2.0, 3.0]
    layer_names = ["Fill", "Sand", "Sandy Clay", "Very Stiff Clay"]
    e_sl_layers = [20, 42, 37.4, 59.5]
    b_layers = [1.0, 0.8, 0.7, 0.6]
    e_ss_layers = [
        e_ss_from_e_sl(e_sl=e_sl, b=b)
        for e_sl, b in zip(e_sl_layers, b_layers, strict=True)
    ]

    soils = [
        Soil(e_sl=e_sl, e_ss=e_ss, soil_name=soil_name)
        for e_sl, e_ss, soil_name in zip(
            e_sl_layers, e_ss_layers, layer_names, strict=True
        )
    ]

    soil_profile = SoilProfile(h_layers=h_layers, soils=soils)

    slab = Slab(
        f_tf=0.7 * (40**0.5),
        thickness=0.200,
    )

    check_slab = CCAA_T48(
        slab=slab,
        loads={"forklift": forklift},
        soil_profile=soil_profile,
        material_factor=MaterialFactor.MIDRANGE,
    )

    assert isclose(check_slab.f_all(load_id="forklift"), 2.15, rel_tol=2e-2)
