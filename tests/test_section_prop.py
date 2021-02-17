"""
Some tests for the section properties file
"""

import pytest

from section_prop import (
    Section,
    GenericSection,
    CombinedSection,
    Polygon,
    Point,
    make_I,
    make_T,
)


ALL_PROPERTIES = [
    "area",
    "Ixx",
    "Iyy",
    "Ixy",
    "Izz",
    "I11",
    "I22",
    "I33",
    "I12",
    "Iuu",
    "Ivv",
    "Iww",
    "Iuv",
    "rxx",
    "ryy",
    "rzz",
    "r11",
    "r22",
    "r33",
    "ruu",
    "rvv",
    "rww",
    "principal_angle",
    "x_c",
    "y_c",
    "x_plus",
    "x_minus",
    "y_plus",
    "y_minus",
    "depth",
    "width",
    "elastic_modulus_uu_plus",
    "elastic_modulus_uu_minus",
    "elastic_modulus_vv_plus",
    "elastic_modulus_vv_minus",
]


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
