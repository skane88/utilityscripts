"""
Some tests for the section properties file
"""

import math

import pandas as pd
import pytest

from utilityscripts.section_prop import (
    CombinedSection,
    GenericSection,
    Point,
    Polygon,
    Rectangle,
    Section,
    make_I,
    make_T,
)

AXIS_INDEPENDENT_PROPERTIES = [
    "area",
    "principal_angle",
    "principal_angle_degrees",
]

GLOBAL_AXIS_PROPERTIES = [
    "Ixx",
    "Iyy",
    "Ixy",
    "Izz",
    "rxx",
    "ryy",
    "rzz",
    "x_c",
    "y_c",
]

LOCAL_AXIS_PROPERTIES = [
    "Iuu",
    "Ivv",
    "Iww",
    "Iuv",
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
    "I11",
    "I22",
    "I33",
    "I12",
    "r11",
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
    "J",
    "J_approx",
    "Iw",
]

ALL_PROPERTIES = (
    AXIS_INDEPENDENT_PROPERTIES
    + GLOBAL_AXIS_PROPERTIES
    + LOCAL_AXIS_PROPERTIES
    + PRINCIPAL_AXIS_PROPERTIES
)


def get_I_sections():

    df = pd.read_excel("steel_sections.xlsx")
    df = df[df["Section Type"] == "I"]  # filter for I sections

    return df.to_dict("records")  # use to_dict() to get each row as a dict.


def I_poly(b_f, d, t_f, t_w):
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

    T = make_T(b_f=100, d=200, t_f=20, t_w=6)
    T = T.move_to_point(origin="origin", end_point=(0, T.extreme_y_minus))

    sections.append((g, T))

    # next make an I section and rotate it by 45deg

    p = I_poly(b_f=100, d=200, t_f=20, t_w=6)
    I1 = GenericSection(poly=p)
    I2 = make_I(b_f=100, d=200, t_f=20, t_w=6)

    I1 = I1.rotate(angle=45, origin="origin", use_radians=False)
    I2 = I2.rotate(angle=45, origin="origin", use_radians=False)

    sections.append((I1, I2))

    combined = make_I(b_f=311, d=327.2, t_f=25, t_w=15.7)
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
    "property",
    ALL_PROPERTIES,
)
@pytest.mark.parametrize("sections", make_sections_for_combined_same_as_generic())
def test_combined_section_gives_same_as_generic(property, sections):
    """
    Test a test shape of a CombinedSection
    """

    assert round(getattr(sections[0], property)) == round(
        getattr(sections[1], property)
    )


def make_sections_for_combined_to_poly_is_correct():

    sections = []

    # first section will be a T offset so its bottom flange is at the 0,0 origin
    p = Polygon(
        [(-50, 0), (-50, 20), (-3, 20), (-3, 200), (3, 200), (3, 20), (50, 20), (50, 0)]
    )
    from_poly = GenericSection(poly=p)

    combined = make_T(b_f=100, d=200, t_f=20, t_w=6)
    combined = combined.move_to_point(
        origin="origin", end_point=(0, combined.extreme_y_minus)
    )

    combined = GenericSection(poly=combined.polygon)

    sections.append((from_poly, combined))

    # next make an I section and rotate it by 45deg

    p = I_poly(b_f=100, d=200, t_f=20, t_w=6)
    from_poly = GenericSection(poly=p)
    combined = make_I(b_f=100, d=200, t_f=20, t_w=6)

    from_poly = from_poly.rotate(angle=45, origin="origin", use_radians=False)
    combined = combined.rotate(angle=45, origin="origin", use_radians=False)
    combined = GenericSection(poly=combined.polygon)

    sections.append((from_poly, combined))

    p = I_poly(b_f=311, d=327.2, t_f=25, t_w=15.7)
    from_poly = GenericSection(poly=p)
    combined = make_I(b_f=311, d=327.2, t_f=25, t_w=15.7)
    combined = GenericSection(poly=combined.polygon)

    sections.append((combined, from_poly))

    return sections


@pytest.mark.parametrize(
    "property",
    ALL_PROPERTIES,
)
@pytest.mark.parametrize("sections", make_sections_for_combined_to_poly_is_correct())
def test_combined_section_to_poly_is_correct(property, sections):
    """
    Test that creating a polygon from a combined section gives the same properties.
    """

    c = make_I(b_f=311, d=327.2, t_f=25, t_w=15.7)
    from_p = GenericSection(poly=c.polygon)

    assert round(getattr(c, property)) == round(getattr(from_p, property))


@pytest.mark.parametrize(
    "property", ["d", "bf", "Ag", "Ix", "Zx", "Sx", "rx", "Iy", "Zy", "Sy", "ry", "J"]
)
@pytest.mark.parametrize("data", get_I_sections(), ids=lambda x: x["Section"])
def test_against_standard_sects(data, property):
    """
    Compare the results of section_prop vs tabulated data for standard AS sections.

    Note that this ignores the radius / fillet welds in the web-flange intersection.
    """

    MAP_SSHEET_TO_SECTION_PROP = {
        "d": ["depth"],
        "bf": ["width"],
        "Ag": ["area"],
        "Ix": ["Ixx", "Iuu", "I11"],
        "Zx": [
            "elastic_modulus_uu_plus",
            "elastic_modulus_uu_minus",
            "elastic_modulus_uu",
            "elastic_modulus_11_plus",
            "elastic_modulus_11_minus",
            "elastic_modulus_11",
        ],
        "Sx": ["plastic_modulus_11"],
        "rx": ["rxx", "ruu", "r11"],
        "Iy": ["Iyy", "Ivv", "I22"],
        "Zy": [
            "elastic_modulus_vv_plus",
            "elastic_modulus_vv_minus",
            "elastic_modulus_vv",
            "elastic_modulus_22_plus",
            "elastic_modulus_22_minus",
            "elastic_modulus_22",
        ],
        "Sy": ["plastic_modulus_22"],
        "ry": ["ryy", "rvv", "r22"],
        "J": ["J_approx"],
    }

    I = make_I(b_f=data["bf"], d=data["d"], t_f=data["tf"], t_w=data["tw"])

    attribute = MAP_SSHEET_TO_SECTION_PROP[property]

    for a in attribute:

        calculated = getattr(I, a)
        test = data[property]

        assert math.isclose(calculated, test, rel_tol=0.1)


@pytest.mark.parametrize(
    "property", ["d", "bf", "Ag", "Ix", "Zx", "Sx", "rx", "Iy", "Zy", "ry", "Sy"]
)
@pytest.mark.parametrize("data", get_I_sections(), ids=lambda x: x["Section"])
def test_against_standard_sects_with_radius(data, property):
    """
    Compare the results of section_prop vs tabulated data for standard AS sections.
    """

    MAP_SSHEET_TO_SECTION_PROP = {
        "d": ["depth"],
        "bf": ["width"],
        "Ag": ["area"],
        "Ix": ["Ixx", "Iuu", "I11"],
        "Zx": [
            "elastic_modulus_uu_plus",
            "elastic_modulus_uu_minus",
            "elastic_modulus_uu",
            "elastic_modulus_11_plus",
            "elastic_modulus_11_minus",
            "elastic_modulus_11",
        ],
        "Sx": ["plastic_modulus_11"],
        "rx": ["rxx", "ruu", "r11"],
        "Iy": ["Iyy", "Ivv", "I22"],
        "Zy": [
            "elastic_modulus_vv_plus",
            "elastic_modulus_vv_minus",
            "elastic_modulus_vv",
            "elastic_modulus_22_plus",
            "elastic_modulus_22_minus",
            "elastic_modulus_22",
        ],
        "Sy": ["plastic_modulus_22"],
        "ry": ["ryy", "rvv", "r22"],
        "J": ["J_approx"],
    }

    if data["Fabrication Type"] == "Hot Rolled":
        I = make_I(
            b_f=data["bf"],
            d=data["d"],
            t_f=data["tf"],
            t_w=data["tw"],
            radius_or_weld="r",
            radius_size=data["r1"],
        )
    else:
        I = make_I(
            b_f=data["bf"],
            d=data["d"],
            t_f=data["tf"],
            t_w=data["tw"],
            radius_or_weld="w",
            weld_size=0.005,  # guestimated weld size
        )

    attribute = MAP_SSHEET_TO_SECTION_PROP[property]

    for a in attribute:

        calculated = getattr(I, a)
        test = data[property]

        assert math.isclose(calculated, test, rel_tol=0.03)


@pytest.mark.parametrize(
    "test_input,cut_height, expected",
    [
        (
            Rectangle(length=100, thickness=10, rotation_angle=90, use_radians=False),
            0,
            12500,
        ),
        (make_I(b_f=100, d=100, t_f=10, t_w=10), 0, 53000),
        (make_I(b_f=100, d=100, t_f=10, t_w=10), 40, 45000),
    ],
)
def test_first_moment_uu(test_input, cut_height, expected):
    """
    Tests for the first moment function.
    """

    actual = test_input.first_moment_uu(cut_height=cut_height)
    assert math.isclose(actual, expected)


@pytest.mark.parametrize(
    "test_input, cut_right, expected",
    [
        (
            Rectangle(length=100, thickness=10, rotation_angle=90, use_radians=False),
            0,
            1250,
        ),
        (make_I(b_f=100, d=100, t_f=10, t_w=10), 0, 26000),
        (make_I(b_f=100, d=100, t_f=10, t_w=10), 40, 9000),
    ],
)
def test_first_moment_vv(test_input, cut_right, expected):
    """
    Tests for the first moment function.
    """

    actual = test_input.first_moment_vv(cut_right=cut_right)
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
            make_I(b_f=100, d=100, t_f=10, t_w=10).rotate(angle=45, use_radians=False),
            0,
            53000,
        ),
        (
            make_I(b_f=100, d=100, t_f=10, t_w=10).rotate(angle=45, use_radians=False),
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
            make_I(b_f=100, d=100, t_f=10, t_w=10).rotate(angle=45, use_radians=False),
            0,
            26000,
        ),
        (
            make_I(b_f=100, d=100, t_f=10, t_w=10).rotate(angle=45, use_radians=False),
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
            make_I(b_f=100, d=100, t_f=10, t_w=10),
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

    sect = make_I(b_f=100, d=100, t_f=10, t_w=10).move(x=10, y=10)

    assert math.isclose(
        sect.first_moment_uu(cut_uu=15), sect.first_moment_uu(cut_xx=25)
    )
    assert math.isclose(
        sect.first_moment_vv(cut_vv=15), sect.first_moment_vv(cut_yy=25)
    )


def test_plastic_modulus_11_rotated():
    """
    Check that the plastic_modulus_11 / 22 functions work even if section is rotated
    """

    T_start = make_T(b_f=100, d=100, t_f=10, t_w=10)

    t_expected_11 = T_start.plastic_modulus_uu
    t_expected_22 = T_start.plastic_modulus_vv

    T_end = T_start.rotate(angle=45, use_radians=False)

    assert math.isclose(T_end.plastic_modulus_11, t_expected_11)
    assert math.isclose(T_end.plastic_modulus_22, t_expected_22)
