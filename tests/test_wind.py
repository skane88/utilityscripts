"""
Contains some tests for the AS1170.2 module.
"""

from math import isclose, log10
from pathlib import Path

import pytest
import toml

from utilityscripts.wind.as1170_2 import (
    FaceType,
    StandardVersion,
    WindRegion,
    WindSite,
    c_pi_open,
    k_a,
    k_v,
    m_d_exact,
    v_r,
    valid_region,
)

FILE_PATH = Path(__file__)
TEST_DATA_PATH_2011 = FILE_PATH.parent / Path("test_as1170_2_2011.toml")
TEST_DATA_2011 = toml.load(TEST_DATA_PATH_2011)

TEST_DATA_PATH_2021 = FILE_PATH.parent / Path("test_as1170_2_2021.toml")
TEST_DATA_2021 = toml.load(TEST_DATA_PATH_2021)


@pytest.mark.parametrize(
    "region, version, expected",
    [
        (WindRegion.A1, StandardVersion.AS1170_2_2011, True),
        (WindRegion.A1, StandardVersion.AS1170_2_2021, True),
        (WindRegion.A0, StandardVersion.AS1170_2_2011, False),
        (WindRegion.A7, StandardVersion.AS1170_2_2021, False),
    ],
)
def test_valid_region(region, version, expected):
    assert valid_region(region, version) == expected


def build_v_r_pairs_2011(v_r_data):
    test_data = TEST_DATA_2011[v_r_data]

    return [
        (WindRegion(region), r, v_r_exp)
        for region, v in test_data.items()
        for r, v_r_exp in v.items()
    ]


@pytest.mark.parametrize(
    "region, r, v_r_exp", build_v_r_pairs_2011("regional_windspeeds_no_F_x")
)
def test_v_r_no_f_x_2011(region, r, v_r_exp):
    """
    Basic test of the V_R method.
    Ignores F_x because that's how the data I have is formatted.
    """

    v_r_calc = round(
        v_r(
            wind_region=region,
            return_period=int(r),
            ignore_m_c=True,
            version=StandardVersion.AS1170_2_2011,
        )
    )  # round because the data from AS1170 is rounded

    assert v_r_calc == v_r_exp


@pytest.mark.parametrize(
    "region, r, v_r_exp", build_v_r_pairs_2011("regional_windspeeds")
)
def test_v_r_2011(region, r, v_r_exp):
    """
    Basic test of the V_R method.
    """

    v_r_calc = v_r(
        wind_region=region,
        return_period=int(r),
        ignore_m_c=False,
        version=StandardVersion.AS1170_2_2011,
    )

    v_r_calc = round(v_r_calc)  # round because the data from AS1170 is rounded

    assert v_r_calc == v_r_exp


def build_v_r_pairs_2021(v_r_data):
    test_data = TEST_DATA_2021[v_r_data]

    return [
        (region, r, v_r_exp)
        for region, v in test_data.items()
        for r, v_r_exp in v.items()
    ]


@pytest.mark.parametrize(
    "region, r, v_r_expected", build_v_r_pairs_2021("regional_windspeeds_no_mc")
)
def test_v_r_2021_no_mc(region, r, v_r_expected):
    """
    Basic test of the V_R method.
    """

    v_r_calc = v_r(
        wind_region=region,
        return_period=int(r),
        ignore_m_c=True,
        version=StandardVersion.AS1170_2_2021,
    )

    v_r_calc = round(v_r_calc)  # round because the data from AS1170 is rounded

    assert v_r_calc == v_r_expected


@pytest.mark.parametrize(
    "region, r, v_r_expected", build_v_r_pairs_2021("regional_windspeeds")
)
def test_v_r_2021(region, r, v_r_expected):
    """
    Basic test of the V_R method.
    """

    v_r_calc = v_r(
        wind_region=region,
        return_period=int(r),
        ignore_m_c=False,
        version=StandardVersion.AS1170_2_2021,
    )

    v_r_calc = round(v_r_calc)  # round because the data from AS1170 is rounded

    assert v_r_calc == v_r_expected


def build_direction_pairs_2011():
    test_data = TEST_DATA_2011["wind_direction_factor"]

    pairs = []

    for region, v in test_data.items():
        for direction, m_d_val in v.items():
            if direction.lower() != "any":
                direction = float(direction)

            m_d_val = tuple(m_d_val)

            pairs.append([(region, direction), m_d_val])

            if isinstance(direction, str):
                continue

            pairs.extend(
                (
                    [(region, direction + 360), m_d_val],
                    [(region, direction - 360), m_d_val],
                )
            )

    return pairs


@pytest.mark.parametrize("input_vals", build_direction_pairs_2011())
def test_m_d_2011(input_vals):
    wind_region = input_vals[0][0]
    direction = input_vals[0][1]
    expected = input_vals[1]

    m_d_calc = m_d_exact(
        wind_region=wind_region,
        direction=direction,
        version=StandardVersion.AS1170_2_2011,
    )

    assert isclose(m_d_calc[0], expected[0])
    assert isclose(m_d_calc[1], expected[1])


def build_direction_pairs_2021():
    test_data = TEST_DATA_2021["wind_direction_factor"]

    pairs = []

    for region, v in test_data.items():
        for direction, m_d_val in v.items():
            if direction.lower() != "any":
                direction = float(direction)

            m_d_val = tuple(m_d_val)

            pairs.append([(region, direction), m_d_val])

            if isinstance(direction, str):
                continue

            pairs.extend(
                (
                    [(region, direction + 360), m_d_val],
                    [(region, direction - 360), m_d_val],
                )
            )

    return pairs


@pytest.mark.parametrize("input_vals", build_direction_pairs_2021())
def test_m_d_2021(input_vals):
    wind_region = input_vals[0][0]
    direction = input_vals[0][1]
    expected = input_vals[1]

    m_d_calc = m_d_exact(
        wind_region=wind_region,
        direction=direction,
        version=StandardVersion.AS1170_2_2021,
    )

    assert isclose(m_d_calc[0], expected[0])
    assert isclose(m_d_calc[1], expected[1])


def test_m_zcat_basic():
    raise AssertionError()


def test_m_zcat_ave():
    raise AssertionError()


def test_m_s():
    raise AssertionError()


@pytest.mark.parametrize(
    "area_ratio, wind_region, governing_face, c_pe, k_a, k_l, open_area, volume, version, expected",
    [
        (
            0.1,
            WindRegion.C,
            FaceType.WINDWARD,
            -0.2,
            1.0,
            1.0,
            100.0,
            1000.0,
            StandardVersion.AS1170_2_2021,
            (-0.20 * 0.70, -0.20 * 0.70),
        ),
        (
            0.1,
            WindRegion.A0,
            FaceType.WINDWARD,
            -0.2,
            1.0,
            1.0,
            100.0,
            1000.0,
            StandardVersion.AS1170_2_2021,
            (-0.30, 0.00),
        ),
        (
            0.1,
            WindRegion.C,
            FaceType.SIDE,
            -0.2,
            1.0,
            1.0,
            100.0,
            1000.0,
            StandardVersion.AS1170_2_2021,
            (-0.20, -0.20),
        ),
        (
            0.1,
            WindRegion.A0,
            FaceType.SIDE,
            -0.2,
            1.0,
            1.0,
            100.0,
            1000.0,
            StandardVersion.AS1170_2_2021,
            (-0.30, 0.00),
        ),
        (
            3.0,
            WindRegion.A0,
            FaceType.WINDWARD,
            0.80,
            1.0,
            1.0,
            100.0,
            1000.0,
            StandardVersion.AS1170_2_2021,
            (0.85 * 0.80, 0.85 * 0.80),
        ),
        (
            1.0,
            WindRegion.A0,
            FaceType.WINDWARD,
            0.80,
            1.0,
            1.0,
            100.0,
            1000.0,
            StandardVersion.AS1170_2_2021,
            (-0.10, 0.20),
        ),
        (
            6.0,
            WindRegion.C,
            FaceType.WINDWARD,
            0.80,
            1.0,
            1.0,
            100.0,
            1000.0,
            StandardVersion.AS1170_2_2021,
            (1.0 * 0.80 * 1.085, 1.0 * 0.80 * 1.085),
        ),
    ],
)
def test_c_pi_open(
    area_ratio,
    wind_region,
    governing_face,
    c_pe,
    k_a,
    k_l,
    open_area,
    volume,
    version,
    expected,
):
    result = c_pi_open(
        area_ratio=area_ratio,
        wind_region=wind_region,
        governing_face=governing_face,
        c_pe=c_pe,
        k_a=k_a,
        k_l=k_l,
        open_area=open_area,
        volume=volume,
        version=version,
    )

    assert isclose(
        result[0],
        expected[0],
    )
    assert isclose(result[1], expected[1])


def test_c_pi_other():
    raise AssertionError()


def k_v_calc(area, volume):
    """
    Helper function to calculate k_v independently of the main code.
    """
    alpha = 100 * (area ** (3 / 2)) / volume
    return 1.01 + 0.15 * log10(alpha)


@pytest.mark.parametrize(
    "area, vol, expected",
    [
        (100, 33_333, 1.085),
        (209, 100_000, 1.085),
        (100, 1, 1.085),
        (100, 1_111_112, 0.85),
        (1, 1_111_112, 0.85),
        (20, 100_000, 0.85),
        (100, 33_334, k_v_calc(100, 33_334)),
        (100, 50_000, k_v_calc(100, 50_000)),
        (100, 100_000, k_v_calc(100, 100_000)),
        (100, 500_000, k_v_calc(100, 500_000)),
        (100, 1_111_111, k_v_calc(100, 1_111_111)),
    ],
)
def test_k_v(area, vol, expected):
    assert isclose(k_v(open_area=area, volume=vol), expected)


@pytest.mark.parametrize(
    "area, face_type, z, version, expected",
    [
        (5, FaceType.SIDE, 10, StandardVersion.AS1170_2_2021, 1.0),
        (5, FaceType.ROOF, 10, StandardVersion.AS1170_2_2021, 1.0),
        (5, FaceType.WINDWARD, 10, StandardVersion.AS1170_2_2021, 1.0),
        (5, FaceType.LEEWARD, 10, StandardVersion.AS1170_2_2021, 1.0),
        (10, FaceType.SIDE, 10, StandardVersion.AS1170_2_2021, 1.0),
        (10, FaceType.ROOF, 10, StandardVersion.AS1170_2_2021, 1.0),
        (10, FaceType.WINDWARD, 10, StandardVersion.AS1170_2_2021, 1.0),
        (10, FaceType.LEEWARD, 10, StandardVersion.AS1170_2_2021, 1.0),
        (25, FaceType.SIDE, 10, StandardVersion.AS1170_2_2021, 0.90),
        (25, FaceType.ROOF, 10, StandardVersion.AS1170_2_2021, 0.90),
        (25, FaceType.WINDWARD, 10, StandardVersion.AS1170_2_2021, 0.95),
        (25, FaceType.LEEWARD, 10, StandardVersion.AS1170_2_2021, 1.0),
        (40, FaceType.SIDE, 10, StandardVersion.AS1170_2_2021, 0.88),
        (40, FaceType.ROOF, 10, StandardVersion.AS1170_2_2021, 0.88),
        (40, FaceType.WINDWARD, 10, StandardVersion.AS1170_2_2021, 0.94),
        (40, FaceType.LEEWARD, 10, StandardVersion.AS1170_2_2021, 0.99),
        (100, FaceType.SIDE, 10, StandardVersion.AS1170_2_2021, 0.80),
        (100, FaceType.ROOF, 10, StandardVersion.AS1170_2_2021, 0.80),
        (100, FaceType.WINDWARD, 10, StandardVersion.AS1170_2_2021, 0.90),
        (100, FaceType.LEEWARD, 10, StandardVersion.AS1170_2_2021, 0.95),
        (5, FaceType.SIDE, 30, StandardVersion.AS1170_2_2021, 1.0),
        (5, FaceType.ROOF, 30, StandardVersion.AS1170_2_2021, 1.0),
        (5, FaceType.WINDWARD, 30, StandardVersion.AS1170_2_2021, 1.0),
        (5, FaceType.LEEWARD, 30, StandardVersion.AS1170_2_2021, 1.0),
        (10, FaceType.SIDE, 30, StandardVersion.AS1170_2_2021, 1.0),
        (10, FaceType.ROOF, 30, StandardVersion.AS1170_2_2021, 1.0),
        (10, FaceType.WINDWARD, 30, StandardVersion.AS1170_2_2021, 1.0),
        (10, FaceType.LEEWARD, 30, StandardVersion.AS1170_2_2021, 1.0),
        (25, FaceType.SIDE, 30, StandardVersion.AS1170_2_2021, 0.90),
        (25, FaceType.ROOF, 30, StandardVersion.AS1170_2_2021, 0.90),
        (25, FaceType.WINDWARD, 30, StandardVersion.AS1170_2_2021, 1.0),
        (25, FaceType.LEEWARD, 30, StandardVersion.AS1170_2_2021, 1.0),
        (40, FaceType.SIDE, 30, StandardVersion.AS1170_2_2021, 0.88),
        (40, FaceType.ROOF, 30, StandardVersion.AS1170_2_2021, 0.88),
        (40, FaceType.WINDWARD, 30, StandardVersion.AS1170_2_2021, 1.00),
        (40, FaceType.LEEWARD, 30, StandardVersion.AS1170_2_2021, 1.00),
        (100, FaceType.SIDE, 30, StandardVersion.AS1170_2_2021, 0.80),
        (100, FaceType.ROOF, 30, StandardVersion.AS1170_2_2021, 0.80),
        (100, FaceType.WINDWARD, 30, StandardVersion.AS1170_2_2021, 1.0),
        (100, FaceType.LEEWARD, 30, StandardVersion.AS1170_2_2021, 1.0),
        (5, FaceType.SIDE, 10, StandardVersion.AS1170_2_2011, 1.0),
        (5, FaceType.ROOF, 10, StandardVersion.AS1170_2_2011, 1.0),
        (5, FaceType.WINDWARD, 10, StandardVersion.AS1170_2_2011, 1.0),
        (5, FaceType.LEEWARD, 10, StandardVersion.AS1170_2_2011, 1.0),
        (10, FaceType.SIDE, 10, StandardVersion.AS1170_2_2011, 1.0),
        (10, FaceType.ROOF, 10, StandardVersion.AS1170_2_2011, 1.0),
        (10, FaceType.WINDWARD, 10, StandardVersion.AS1170_2_2011, 1.0),
        (10, FaceType.LEEWARD, 10, StandardVersion.AS1170_2_2011, 1.0),
        (25, FaceType.SIDE, 10, StandardVersion.AS1170_2_2011, 0.90),
        (25, FaceType.ROOF, 10, StandardVersion.AS1170_2_2011, 0.90),
        (25, FaceType.WINDWARD, 10, StandardVersion.AS1170_2_2011, 1.00),
        (25, FaceType.LEEWARD, 10, StandardVersion.AS1170_2_2011, 1.0),
        (40, FaceType.SIDE, 10, StandardVersion.AS1170_2_2011, 0.88),
        (40, FaceType.ROOF, 10, StandardVersion.AS1170_2_2011, 0.88),
        (40, FaceType.WINDWARD, 10, StandardVersion.AS1170_2_2011, 1.00),
        (40, FaceType.LEEWARD, 10, StandardVersion.AS1170_2_2011, 1.00),
        (100, FaceType.SIDE, 10, StandardVersion.AS1170_2_2011, 0.80),
        (100, FaceType.ROOF, 10, StandardVersion.AS1170_2_2011, 0.80),
        (100, FaceType.WINDWARD, 10, StandardVersion.AS1170_2_2011, 1.00),
        (100, FaceType.LEEWARD, 10, StandardVersion.AS1170_2_2011, 1.00),
        (5, FaceType.SIDE, 30, StandardVersion.AS1170_2_2011, 1.0),
        (5, FaceType.ROOF, 30, StandardVersion.AS1170_2_2011, 1.0),
        (5, FaceType.WINDWARD, 30, StandardVersion.AS1170_2_2011, 1.0),
        (5, FaceType.LEEWARD, 30, StandardVersion.AS1170_2_2011, 1.0),
        (10, FaceType.SIDE, 30, StandardVersion.AS1170_2_2011, 1.0),
        (10, FaceType.ROOF, 30, StandardVersion.AS1170_2_2011, 1.0),
        (10, FaceType.WINDWARD, 30, StandardVersion.AS1170_2_2011, 1.0),
        (10, FaceType.LEEWARD, 30, StandardVersion.AS1170_2_2011, 1.0),
        (25, FaceType.SIDE, 30, StandardVersion.AS1170_2_2011, 0.90),
        (25, FaceType.ROOF, 30, StandardVersion.AS1170_2_2011, 0.90),
        (25, FaceType.WINDWARD, 30, StandardVersion.AS1170_2_2011, 1.0),
        (25, FaceType.LEEWARD, 30, StandardVersion.AS1170_2_2011, 1.0),
        (40, FaceType.SIDE, 30, StandardVersion.AS1170_2_2011, 0.88),
        (40, FaceType.ROOF, 30, StandardVersion.AS1170_2_2011, 0.88),
        (40, FaceType.WINDWARD, 30, StandardVersion.AS1170_2_2011, 1.00),
        (40, FaceType.LEEWARD, 30, StandardVersion.AS1170_2_2011, 1.00),
        (100, FaceType.SIDE, 30, StandardVersion.AS1170_2_2011, 0.80),
        (100, FaceType.ROOF, 30, StandardVersion.AS1170_2_2011, 0.80),
        (100, FaceType.WINDWARD, 30, StandardVersion.AS1170_2_2011, 1.0),
        (100, FaceType.LEEWARD, 30, StandardVersion.AS1170_2_2011, 1.0),
    ],
)
def test_k_a(area, face_type, z, version, expected):
    assert isclose(k_a(area=area, face_type=face_type, z=z, version=version), expected)


def test_windsite():
    """
    Basic test of the WindSite class.
    """

    site = WindSite(
        wind_region=WindRegion.A1, terrain_category=1.0, shielding_data=None
    )

    assert site.wind_region == WindRegion.A1
    assert site.terrain_category == 1.0
    assert site.shielding_data is None
