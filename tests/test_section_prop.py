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


def make_sections():

    sections = []

    # first section will be a T offset so its bottom flange is at the 0,0 origin
    p = Polygon(
        [(-50, 0), (-50, 20), (-3, 20), (-3, 200), (3, 200), (3, 20), (50, 20), (50, 0)]
    )
    g = GenericSection(poly=p)

    T = make_T(b_f=100, d=200, t_f=20, t_w=6)
    T = T.move_to_point(origin="origin", end_point=(0, T.y_minus))

    sections.append((g, T))

    # next make an I section and rotate it by 45deg

    p = Polygon(
        [
            (-50, -100),
            (-50, -80),
            (-3, -80),
            (-3, 80),
            (-50, 80),
            (-50, 100),
            (50, 100),
            (50, 80),
            (3, 80),
            (3, -80),
            (50, -80),
            (50, -100),
        ]
    )
    I1 = GenericSection(poly=p)
    I2 = make_I(b_f=100, d=200, t_f=20, t_w=6)

    I1 = I1.rotate(angle=45, origin="origin", use_radians=False)
    I2 = I2.rotate(angle=45, origin="origin", use_radians=False)

    sections.append((I1, I2))

    return sections


@pytest.mark.parametrize(
    "property",
    [
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
        "elastic_modulus_xx_plus",
        "elastic_modulus_xx_minus",
        "elastic_modulus_yy_plus",
        "elastic_modulus_yy_minus",
    ],
)
@pytest.mark.parametrize("sections", make_sections())
def test_combined_section_1(property, sections):
    """
    Test a test shape of a CombinedSection
    """

    assert round(getattr(sections[0], property)) == round(
        getattr(sections[1], property)
    )
