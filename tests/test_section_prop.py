"""
Some tests for the section properties file
"""

import math

import pandas as pd
import pytest

from utilityscripts.section_prop import (
    CombinedSection,
    GenericSection,
    Polygon,
    Rectangle,
    make_i,
    make_t,
)

AXIS_INDEPENDENT_PROPERTIES = [
    "area",
    "principal_angle",
    "principal_angle_degrees",
]

GLOBAL_AXIS_PROPERTIES = [
    "i_xx",
    "i_yy",
    "i_xy",
    "i_zz",
    "rxx",
    "ryy",
    "rzz",
    "x_c",
    "y_c",
]

LOCAL_AXIS_PROPERTIES = [
    "i_uu",
    "i_vv",
    "i_ww",
    "i_uv",
    "ruu",
    "rvv",
    "rww",
    "extreme_x_plus",
    "extreme_x_minus",
    "extreme_y_plus",
    "extreme_y_minus",
    "depth",
    "width",
    "elastic_modulus_uu_plus",
    "elastic_modulus_uu_minus",
    "elastic_modulus_vv_plus",
    "elastic_modulus_vv_minus",
    "plastic_modulus_uu",
    "plastic_modulus_vv",
]

PRINCIPAL_AXIS_PROPERTIES = [
    "i_11",
    "i_22",
    "i_33",
    "i_12",
    "i_11",
    "r22",
    "r33",
    "extreme_11_plus",
    "extreme_11_minus",
    "extreme_22_plus",
    "extreme_22_minus",
    "elastic_modulus_11_plus",
    "elastic_modulus_11_minus",
    "elastic_modulus_22_plus",
    "elastic_modulus_22_minus",
    "plastic_modulus_11",
    "plastic_modulus_22",
    "j",
    "j_approx",
    "i_w",
]

ALL_PROPERTIES = (
    AXIS_INDEPENDENT_PROPERTIES
    + GLOBAL_AXIS_PROPERTIES
    + LOCAL_AXIS_PROPERTIES
    + PRINCIPAL_AXIS_PROPERTIES
)


def get_i_sections():
    section_df = pd.read_excel("..\\data\\steel_data.xlsx", sheet_name="Is")
    section_df = section_df[section_df["section_type"] == "I"]  # filter for I sections

    return section_df.to_dict("records")  # use to_dict() to get each row as a dict.


def i_poly(b_f, d, t_f, t_w):
    """
    Make a polygon I section. Currently has equal flange thicknesses and widths.

    :param b_f: Flange width.
    :param d: Total depth.
    :param t_f: Thickness of flange.
    :param t_w: Thickness of web.
    :return: A Shapely Polygon
    """

    return Polygon(
        [
            (-b_f / 2, -d / 2),
            (-b_f / 2, -d / 2 + t_f),
            (-t_w / 2, -d / 2 + t_f),
            (-t_w / 2, d / 2 - t_f),
            (-b_f / 2, d / 2 - t_f),
            (-b_f / 2, d / 2),
            (b_f / 2, d / 2),
            (b_f / 2, d / 2 - t_f),
            (t_w / 2, d / 2 - t_f),
            (t_w / 2, -d / 2 + t_f),
            (b_f / 2, -d / 2 + t_f),
            (b_f / 2, -d / 2),
        ]
    )


def make_sections_for_combined_same_as_generic():
    sections = []

    # first section will be a T offset so its bottom flange is at the 0,0 origin
    p = Polygon(
        [(-50, 0), (-50, 20), (-3, 20), (-3, 200), (3, 200), (3, 20), (50, 20), (50, 0)]
    )
    g = GenericSection(poly=p)

    t = make_t(b_f=100, d=200, t_f=20, t_w=6)
    t = t.move_to_point(origin="origin", end_point=(0, t.extreme_y_minus))

    sections.append((g, t))

    # next make an I section and rotate it by 45deg

    p = i_poly(b_f=100, d=200, t_f=20, t_w=6)
    i1 = GenericSection(poly=p)
    i2 = make_i(b_f=100, d=200, t_f=20, t_w=6)

    i1 = i1.rotate(angle=45, origin="origin", use_radians=False)
    i2 = i2.rotate(angle=45, origin="origin", use_radians=False)

    sections.append((i1, i2))

    combined = make_i(b_f=311, d=327.2, t_f=25, t_w=15.7)
    from_p = GenericSection(poly=combined.polygon)

    sections.append((combined, from_p))

    return sections


@pytest.mark.parametrize("sections", make_sections_for_combined_same_as_generic())
def test_combined_section_and_generic_centroid(sections):
    a = sections[1].move(x=23, y=23)
    b = sections[0].move(x=23, y=23)

    assert math.isclose(a.centroid.x, b.centroid.x)
    assert math.isclose(a.centroid.y, b.centroid.y)


@pytest.mark.parametrize(
    "sect_property",
    ALL_PROPERTIES,
)
@pytest.mark.parametrize("sections", make_sections_for_combined_same_as_generic())
def test_combined_section_gives_same_as_generic(sect_property, sections):
    """
    Test a test shape of a CombinedSection
    """

    assert round(getattr(sections[0], sect_property)) == round(
        getattr(sections[1], sect_property)
    )


def make_sections_for_combined_to_poly_is_correct():
    sections = []

    # first section will be a T offset so its bottom flange is at the 0,0 origin
    p = Polygon(
        [(-50, 0), (-50, 20), (-3, 20), (-3, 200), (3, 200), (3, 20), (50, 20), (50, 0)]
    )
    from_poly = GenericSection(poly=p)

    combined = make_t(b_f=100, d=200, t_f=20, t_w=6)
    combined = combined.move_to_point(
        origin="origin", end_point=(0, combined.extreme_y_minus)
    )

    combined = GenericSection(poly=combined.polygon)

    sections.append((from_poly, combined))

    # next make an I section and rotate it by 45deg

    p = i_poly(b_f=100, d=200, t_f=20, t_w=6)
    from_poly = GenericSection(poly=p)
    combined = make_i(b_f=100, d=200, t_f=20, t_w=6)

    from_poly = from_poly.rotate(angle=45, origin="origin", use_radians=False)
    combined = combined.rotate(angle=45, origin="origin", use_radians=False)
    combined = GenericSection(poly=combined.polygon)

    sections.append((from_poly, combined))

    p = i_poly(b_f=311, d=327.2, t_f=25, t_w=15.7)
    from_poly = GenericSection(poly=p)
    combined = make_i(b_f=311, d=327.2, t_f=25, t_w=15.7)
    combined = GenericSection(poly=combined.polygon)

    sections.append((combined, from_poly))

    return sections


@pytest.mark.parametrize(
    "sect_property",
    ALL_PROPERTIES,
)
@pytest.mark.parametrize("sections", make_sections_for_combined_to_poly_is_correct())
def test_combined_section_to_poly_is_correct(sect_property, sections):
    """
    Test that creating a polygon from a combined section gives the same properties.
    """

    c = make_i(b_f=311, d=327.2, t_f=25, t_w=15.7)
    from_p = GenericSection(poly=c.polygon)

    assert round(getattr(c, sect_property)) == round(getattr(from_p, sect_property))


@pytest.mark.parametrize(
    "sect_property",
    ["d", "b_f", "a_g", "i_x", "z_x", "s_x", "r_x", "i_y", "z_y", "s_y", "r_y", "j"],
)
@pytest.mark.parametrize("data", get_i_sections(), ids=lambda x: x["section"])
def test_against_standard_sects(data, sect_property):
    """
    Compare the results of section_prop vs tabulated data for standard AS sections.

    Note that this ignores the radius / fillet welds in the web-flange intersection.
    """

    map_ssheet_to_section_prop = {
        "d": ["depth"],
        "b_f": ["width"],
        "a_g": ["area"],
        "i_x": ["i_xx", "i_uu", "i_11"],
        "z_x": [
            "elastic_modulus_uu_plus",
            "elastic_modulus_uu_minus",
            "elastic_modulus_uu",
            "elastic_modulus_11_plus",
            "elastic_modulus_11_minus",
            "elastic_modulus_11",
        ],
        "s_x": ["plastic_modulus_11"],
        "r_x": ["rxx", "ruu", "r11"],
        "i_y": ["i_yy", "i_vv", "i_22"],
        "z_y": [
            "elastic_modulus_vv_plus",
            "elastic_modulus_vv_minus",
            "elastic_modulus_vv",
            "elastic_modulus_22_plus",
            "elastic_modulus_22_minus",
            "elastic_modulus_22",
        ],
        "s_y": ["plastic_modulus_22"],
        "r_y": ["ryy", "rvv", "r22"],
        "j": ["j_approx"],
    }

    i_sect = make_i(b_f=data["b_f"], d=data["d"], t_f=data["t_f"], t_w=data["t_w"])

    attribute = map_ssheet_to_section_prop[sect_property]

    for a in attribute:
        calculated = getattr(i_sect, a)
        test = data[sect_property]

        assert math.isclose(calculated, test, rel_tol=0.1)


@pytest.mark.parametrize(
    "sect_property",
    ["d", "b_f", "a_g", "i_x", "z_x", "s_x", "r_x", "i_y", "z_y", "r_y", "s_y"],
)
@pytest.mark.parametrize("data", get_i_sections(), ids=lambda x: x["section"])
def test_against_standard_sects_with_radius(data, sect_property):
    """
    Compare the results of section_prop vs tabulated data for standard AS sections.
    """

    map_ssheet_to_section_prop = {
        "d": ["depth"],
        "b_f": ["width"],
        "a_g": ["area"],
        "i_x": ["i_xx", "i_uu", "i_11"],
        "z_x": [
            "elastic_modulus_uu_plus",
            "elastic_modulus_uu_minus",
            "elastic_modulus_uu",
            "elastic_modulus_11_plus",
            "elastic_modulus_11_minus",
            "elastic_modulus_11",
        ],
        "s_x": ["plastic_modulus_11"],
        "r_x": ["rxx", "ruu", "r11"],
        "i_y": ["i_yy", "i_vv", "i_22"],
        "z_y": [
            "elastic_modulus_vv_plus",
            "elastic_modulus_vv_minus",
            "elastic_modulus_vv",
            "elastic_modulus_22_plus",
            "elastic_modulus_22_minus",
            "elastic_modulus_22",
        ],
        "s_y": ["plastic_modulus_22"],
        "r_y": ["ryy", "rvv", "r22"],
        "j": ["j_approx"],
    }

    if data["fabrication_type"] == "Hot Rolled":
        i_sect = make_i(
            b_f=data["b_f"],
            d=data["d"],
            t_f=data["t_f"],
            t_w=data["t_w"],
            radius_or_weld="r",
            radius_size=data["r_1"],
        )
    else:
        i_sect = make_i(
            b_f=data["b_f"],
            d=data["d"],
            t_f=data["t_f"],
            t_w=data["t_w"],
            radius_or_weld="w",
            weld_size=data["w_1"],
        )

    attribute = map_ssheet_to_section_prop[sect_property]

    for a in attribute:
        calculated = getattr(i_sect, a)
        test = data[sect_property]

        assert math.isclose(calculated, test, rel_tol=0.03)


@pytest.mark.parametrize(
    "test_input,cut_height, expected",
    [
        (
            Rectangle(length=100, thickness=10, rotation_angle=90, use_radians=False),
            0,
            12500,
        ),
        (make_i(b_f=100, d=100, t_f=10, t_w=10), 0, 53000),
        (make_i(b_f=100, d=100, t_f=10, t_w=10), 40, 45000),
    ],
)
def test_first_moment_uu(test_input, cut_height, expected):
    """
    Tests for the first moment function.
    """

    actual = test_input.first_moment_uu(cut_uu=cut_height)
    assert math.isclose(actual, expected)


@pytest.mark.parametrize(
    "test_input, cut_right, expected",
    [
        (
            Rectangle(length=100, thickness=10, rotation_angle=90, use_radians=False),
            0,
            1250,
        ),
        (make_i(b_f=100, d=100, t_f=10, t_w=10), 0, 26000),
        (make_i(b_f=100, d=100, t_f=10, t_w=10), 40, 9000),
    ],
)
def test_first_moment_vv(test_input, cut_right, expected):
    """
    Tests for the first moment function.
    """

    actual = test_input.first_moment_vv(cut_vv=cut_right)
    assert math.isclose(actual, expected)


@pytest.mark.parametrize(
    "test_input,cut_22, expected",
    [
        (
            Rectangle(length=100, thickness=10, rotation_angle=45, use_radians=False),
            0,
            12500,
        ),
        (
            make_i(b_f=100, d=100, t_f=10, t_w=10).rotate(angle=45, use_radians=False),
            0,
            53000,
        ),
        (
            make_i(b_f=100, d=100, t_f=10, t_w=10).rotate(angle=45, use_radians=False),
            40,
            45000,
        ),
    ],
)
def test_first_moment_11(test_input, cut_22, expected):
    """
    Tests for the first moment function.
    """

    actual = test_input.first_moment_11(cut_22=cut_22)
    assert math.isclose(actual, expected)


@pytest.mark.parametrize(
    "test_input, cut_11, expected",
    [
        (
            Rectangle(length=100, thickness=10, rotation_angle=45, use_radians=False),
            0,
            1250,
        ),
        (
            make_i(b_f=100, d=100, t_f=10, t_w=10).rotate(angle=45, use_radians=False),
            0,
            26000,
        ),
        (
            make_i(b_f=100, d=100, t_f=10, t_w=10).rotate(angle=45, use_radians=False),
            40,
            9000,
        ),
    ],
)
def test_first_moment_22(test_input, cut_11, expected):
    """
    Tests for the first moment function.
    """

    actual = test_input.first_moment_22(cut_11=cut_11)
    assert math.isclose(actual, expected)


@pytest.mark.parametrize(
    "sect_a,sect_b",
    [
        (
            make_i(b_f=100, d=100, t_f=10, t_w=10),
            GenericSection(
                [
                    (-50, -50),
                    (50, -50),
                    (50, -40),
                    (5, -40),
                    (5, 40),
                    (50, 40),
                    (50, 50),
                    (-50, 50),
                    (-50, 40),
                    (-5, 40),
                    (-5, -40),
                    (-50, -40),
                ]
            ),
        )
    ],
)
def test_plastic_modulus_same(sect_a, sect_b):
    """
    Test the plastic modulus determined about an axis is the same regardless of which
    direction it is calculated and whether it is calculated from a CombinedSection
    or a GenericSection

    :param sect_a: The first section.
    :param sect_b: The second section.
    """

    assert math.isclose(
        sect_a.first_moment_11(cut_22=15, above=True),
        sect_a.first_moment_11(cut_22=15, above=False),
    )
    assert math.isclose(
        sect_b.first_moment_11(cut_22=15, above=True),
        sect_b.first_moment_11(cut_22=15, above=False),
    )
    assert math.isclose(
        sect_a.first_moment_11(cut_22=15, above=True),
        sect_b.first_moment_11(cut_22=15, above=True),
    )
    assert math.isclose(
        sect_a.first_moment_11(cut_22=15, above=False),
        sect_b.first_moment_11(cut_22=15, above=False),
    )

    assert math.isclose(
        sect_a.first_moment_22(cut_11=15, right=True),
        sect_a.first_moment_22(cut_11=15, right=False),
    )
    assert math.isclose(
        sect_b.first_moment_22(cut_11=15, right=True),
        sect_b.first_moment_22(cut_11=15, right=False),
    )
    assert math.isclose(
        sect_a.first_moment_22(cut_11=15, right=True),
        sect_b.first_moment_22(cut_11=15, right=True),
    )
    assert math.isclose(
        sect_a.first_moment_22(cut_11=15, right=False),
        sect_b.first_moment_22(cut_11=15, right=False),
    )


def test_plastic_modulus_uuvv():
    """
    Test the first_moment_uu and first_moment_vv methods are the same whether cutting
    based on the u-u / v-v axes or the x-x / y-y axes.
    """

    sect = make_i(b_f=100, d=100, t_f=10, t_w=10).move(x=10, y=10)

    assert math.isclose(
        sect.first_moment_uu(cut_uu=15), sect.first_moment_uu(cut_xx=25)
    )
    assert math.isclose(
        sect.first_moment_vv(cut_vv=15), sect.first_moment_vv(cut_yy=25)
    )


def test_plastic_modulus_uuvv2():
    i_sect = make_i(
        b_f=0.1905,
        d=0.6096,
        t_f=0.0256794,
        t_w=0.014478,
        radius_or_weld="r",
        radius_size=0.018542,
    ).move(x=0, y=0.6096 / 2 + 0.025)
    plate = Rectangle(length=0.21, thickness=0.025).move(x=0, y=0.025 / 2)
    strengthened = CombinedSection([(i_sect, (0, 0)), (plate, (0, 0))])

    uu = strengthened.first_moment_uu(cut_uu=-(strengthened.centroid.y - 0.025))
    xx = strengthened.first_moment_uu(cut_xx=0.025)

    assert math.isclose(uu, xx)


def test_plastic_modulus_uu3():
    i_sect = make_i(
        b_f=0.1905,
        d=0.6096,
        t_f=0.0256794,
        t_w=0.014478,
        radius_or_weld="r",
        radius_size=0.018542,
    ).move(x=0, y=0.6096 / 2 + 0.025)
    plate = Rectangle(length=0.21, thickness=0.025).move(x=0, y=0.025 / 2)
    strengthened = CombinedSection([(i_sect, (0, 0)), (plate, (0, 0))])

    uu = strengthened.first_moment_uu(cut_xx=0)

    assert math.isclose(uu, 0.0, abs_tol=1e-6)


def test_plastic_modulus_11_rotated():
    """
    Check that the plastic_modulus_11 / 22 functions work even if section is rotated
    """

    t_start = make_t(b_f=100, d=100, t_f=10, t_w=10)

    t_expected_11 = t_start.plastic_modulus_uu
    t_expected_22 = t_start.plastic_modulus_vv

    t_end = t_start.rotate(angle=45, use_radians=False)

    assert math.isclose(t_end.plastic_modulus_11, t_expected_11, abs_tol=1e-6)
    assert math.isclose(t_end.plastic_modulus_22, t_expected_22, abs_tol=1e-6)
