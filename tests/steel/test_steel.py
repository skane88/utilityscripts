"""
Module to contain tests for code in steel.py
"""

from math import isclose

import numpy as np

from utilityscripts.steel.steel import (
    BoltGrade,
    angle_section_df,
    bolt_grades,
    c_section_df,
    chs_section_df,
    i_section_df,
    local_thickness_reqd,
    nearest_standard_plate,
    nearest_standard_weld,
    rhs_section_df,
    standard_grades,
    steel_grades,
)


def test_standard_grades():
    assert standard_grades()


def test_thickness_reqd():
    """
    Test the method that calculates the thickness required for a plate with a point load.
    """

    force = 10e3
    f_y = 250e6
    phi = 0.9

    assert local_thickness_reqd(
        point_force=force, f_y=f_y, phi=phi
    ) == local_thickness_reqd(
        point_force=force, f_y=f_y, phi=phi, width_lever=(0.0, 0.0)
    )

    assert (
        local_thickness_reqd(point_force=force, f_y=f_y, phi=phi)
        == 0.009428090415820633  # noqa: PLR2004
    )

    width_lever = (0.022, 0.035)

    assert (
        local_thickness_reqd(
            point_force=force, f_y=f_y, phi=phi, width_lever=width_lever
        )
        == 0.008223919396586149  # noqa: PLR2004
    )


def test_steel_grades():
    """Test the steel grades function"""

    sg = steel_grades()

    assert sg

    assert sg["AS/NZS3678:250"].get_f_y(0.008) == 280e6  # noqa: PLR2004
    assert sg["AS/NZS3678:250"].get_f_u(0.008) == 410e6  # noqa: PLR2004


def test_nearest_standard_weld():
    """Test the nearest standard weld function"""

    assert nearest_standard_weld(0.006) == 0.006  # noqa: PLR2004
    assert nearest_standard_weld(0.0065, greater_than=False) == 0.006  # noqa: PLR2004
    assert nearest_standard_weld(0.0065, greater_than=True) == 0.008  # noqa: PLR2004


def test_nearest_standard_plate():
    """Test the nearest standard plate function"""

    assert nearest_standard_plate(0.006) == 0.006  # noqa: PLR2004
    assert nearest_standard_plate(0.0065, greater_than=False) == 0.006  # noqa: PLR2004
    assert nearest_standard_plate(0.0065, greater_than=True) == 0.008  # noqa: PLR2004


def test_i_section_df():
    """Test the i_section_df function"""

    i_sects = i_section_df()

    assert not i_sects.is_empty()


def test_c_section_df():
    """Test the c_section_df function"""
    c_sects = c_section_df()

    assert not c_sects.is_empty()


def test_angle_section_df():
    """Test the angle_section_df function"""
    angle_sects = angle_section_df()

    assert not angle_sects.is_empty()


def test_rhs_section_df():
    """Test the rhs_section_df function"""
    rhs_sects = rhs_section_df()

    assert not rhs_sects.is_empty()


def test_chs_section_df():
    """Test the chs_section_df function"""
    chs_sects = chs_section_df()

    assert not chs_sects.is_empty()


def test_bolt_grade():
    """
    Test the bolt grade class
    """

    grade = "8.8"

    thicknesses = [0.000, 0.015999, 0.016, 0.100]
    f_yf = [640e6, 640e6, 660e6, 660e6]
    f_uf = [800e6, 800e6, 830e6, 830e6]
    k_rd = [1.00, 1.00, 1.00, 1.00]

    bg = BoltGrade(grade=grade, diameters=thicknesses, f_yf=f_yf, f_uf=f_uf, k_rd=k_rd)

    thicknesses = np.asarray(thicknesses)
    f_yf = np.asarray(f_yf)
    f_uf = np.asarray(f_uf)

    assert bg.grade == grade
    assert np.array_equal(bg.diameters, thicknesses)
    assert np.array_equal(bg.f_yf, f_yf)
    assert np.array_equal(bg.f_uf, f_uf)

    assert isclose(bg.get_f_yf(0.008), 640e6)
    assert isclose(bg.get_f_yf(0.015999), 640e6)
    assert isclose(bg.get_f_yf(0.016), 660e6)
    assert isclose(bg.get_f_yf(0.099), 660e6)

    assert isclose(bg.get_f_uf(0.008), 800e6)
    assert isclose(bg.get_f_uf(0.015999), 800e6)
    assert isclose(bg.get_f_uf(0.016), 830e6)
    assert isclose(bg.get_f_uf(0.100), 830e6)

    assert isclose(bg.get_k_rd(0.008), 1.00)
    assert isclose(bg.get_k_rd(0.050), 1.00)


def test_bolt_grades():
    """
    Can the bolt_grades method actually initiate grades?
    """

    assert bolt_grades()
    assert isclose(bolt_grades()["8.8"].get_f_yf(0.008), 640e6)
    assert isclose(bolt_grades()["8.8"].get_f_uf(0.008), 800e6)
    assert isclose(bolt_grades()["8.8"].get_f_yf(0.016), 660e6)
    assert isclose(bolt_grades()["8.8"].get_f_uf(0.016), 830e6)
    assert isclose(bolt_grades()["4.6"].get_f_yf(0.008), 240e6)
    assert isclose(bolt_grades()["4.6"].get_f_uf(0.008), 400e6)
    assert isclose(bolt_grades()["4.6"].get_f_yf(0.016), 240e6)
    assert isclose(bolt_grades()["4.6"].get_f_uf(0.016), 400e6)
    assert isclose(bolt_grades()["10.9"].get_f_yf(0.008), 940e6)
    assert isclose(bolt_grades()["10.9"].get_f_uf(0.008), 1040e6)
    assert isclose(bolt_grades()["10.9"].get_f_yf(0.016), 940e6)
    assert isclose(bolt_grades()["10.9"].get_f_uf(0.016), 1040e6)


def test_bolt_grades_krd():
    assert bolt_grades()
    assert isclose(bolt_grades()["8.8"].get_k_rd(0.008), 1.00)
    assert isclose(bolt_grades()["4.6"].get_k_rd(0.016), 1.00)
    assert isclose(bolt_grades()["10.9"].get_k_rd(0.016), 0.83)
