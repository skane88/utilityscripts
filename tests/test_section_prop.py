"""
Some tests for the section properties file
"""

import pytest
import pandas as pd

from section_prop import (
    Section,
    GenericSection,
    CombinedSection,
    Polygon,
    Point,
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


@pytest.mark.parametrize(
    "property", ALL_PROPERTIES,
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
    "property", ALL_PROPERTIES,
)
@pytest.mark.parametrize("sections", make_sections_for_combined_to_poly_is_correct())
def test_combined_section_to_poly_is_correct(property, sections):
    """
    Test that creating a polygon from a combined section gives the same properties.
    """

    c = make_I(b_f=311, d=327.2, t_f=25, t_w=15.7)
    from_p = GenericSection(poly=c.polygon)

    assert round(getattr(c, property)) == round(getattr(from_p, property))


@pytest.mark.parametrize("data", get_I_sections(), ids=lambda x: x["Section"])
def test_against_standard_sects(data):
    """
    Compare the results of section_prop vs tabulated data for standard AS sections.
    """

    I = make_I(b_f=data["bf"], d=data["d"], t_f=data["tf"], t_w=data["tw"])

    assert False
