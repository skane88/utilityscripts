"""
Module to contain tests for code in steel.py
"""

from math import isclose

import pytest

from utilityscripts.steel.as4100 import k_t
from utilityscripts.steel.steel import (
    angle_section_df,
    c_section_df,
    chs_section_df,
    flat_plate_bending_point,
    flat_plate_bending_uniform,
    i_section_df,
    local_thickness_reqd,
    nearest_standard_plate,
    nearest_standard_weld,
    rhs_section_df,
    steel_grades,
)


@pytest.mark.parametrize(
    "restraint_code, d_1, length, t_f, t_w, n_w, expected",
    [
        ("PP", 0.2060, 4.00, 0.0120, 0.0065, 1, 1.081012),
        ("PF", 0.2834, 10.0, 0.0118, 0.0067, 1, 1.019352),
        ("FF", 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
        ("FL", 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
        ("LF", 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
        ("LL", 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
        ("FU", 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
        ("UF", 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
    ],
)
def test_k_t(restraint_code, d_1, length, t_f, t_w, n_w, expected):
    """
    Test the k_t method.
    """

    k_t_calc = k_t(
        restraint_code=restraint_code, d_1=d_1, length=length, t_f=t_f, t_w=t_w, n_w=n_w
    )

    assert isclose(k_t_calc, expected, rel_tol=1e-6)


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


@pytest.mark.parametrize(
    "a, b, t, e, r_o, w, sigma_exp, y_max_exp",
    [
        # Test case from RFEM comparison
        (
            1.0,
            1.0,
            0.008,
            200e9,  # E - Young's modulus for steel
            0.05,  # r_o - radius of load
            10000,  # W - 10 kN load
            270.061e6,  # sigma_exp - from RFEM
            0.0126,  # y_max_exp - from RFEM
        ),
        (
            2.0,
            1.0,
            0.008,
            200e9,  # E - Young's modulus for steel
            0.05,  # r_o - radius of load
            10000,  # W - 10 kN load
            288.061e6,  # sigma_exp - from RFEM
            0.0180,  # y_max_exp - from RFEM
        ),
        (
            4.0,
            1.0,
            0.008,
            200e9,  # E - Young's modulus for steel
            0.05,  # r_o - radius of load
            10000,  # W - 10 kN load
            289.435e6,  # sigma_exp - from RFEM
            0.0184,  # y_max_exp - from RFEM
        ),
    ],
)
def test_flat_plate_bending_point(a, b, t, e, r_o, w, sigma_exp, y_max_exp):
    """Test flat plate bending under point load against RFEM results"""

    tol = 0.10  # 10% tolerance for FEA comparison
    result = flat_plate_bending_point(a=a, b=b, t=t, e=e, r_o=r_o, w=w)

    # Check against FEA results
    assert isclose(result["sigma_max"], sigma_exp, rel_tol=tol)
    assert isclose(result["y_max"], y_max_exp, rel_tol=tol)


@pytest.mark.parametrize(
    "a, b, t, e, q, sigma_exp, y_max_exp, r_max_exp, tol",
    [
        (
            1.0,
            1.0,
            0.008,
            200e9,  # E - Young's modulus for steel
            10000,  # q - 10 kPa uniform load
            43.277e6,  # sigma_exp
            0.0045,  # y_max_exp
            4265,  # R_max_exp
            0.05,  # tol
        ),
        (
            2.0,
            1.0,
            0.008,
            200e9,  # E - Young's modulus for steel
            10000,  # q - 10 kPa uniform load
            82.375e6,  # sigma_exp, calculated in RFEM
            0.0111,  # y_max_exp, calculated in RFEM
            5062,  # R_max_exp, calculated in RFEM
            0.15,  # tol - quite large, but the hand calc overestimates the stress
        ),
        (
            4.0,
            1.0,
            0.008,
            200e9,  # E - Young's modulus for steel
            10000,  # q - 10 kPa uniform load
            103.961e6,  # sigma_exp, calculated in RFEM
            0.0141,  # y_max_exp, calculated in RFEM
            5038,  # R_max_exp, calculated in RFEM
            0.125,  # tol - quite large, but the hand calc overestimates the stress
        ),
    ],
)
def test_flat_plate_bending_uniform(
    a, b, t, e, q, sigma_exp, y_max_exp, r_max_exp, tol
):
    """Test flat plate bending under uniform load"""

    result = flat_plate_bending_uniform(a=a, b=b, t=t, e=e, q=q)

    assert isclose(result["sigma_max"], sigma_exp, rel_tol=tol)
    assert isclose(result["y_max"], y_max_exp, rel_tol=tol)
    assert isclose(result["R_max"], r_max_exp, rel_tol=tol)


def test_steel_grades():
    """Test the steel grades function"""

    sg = steel_grades()

    assert sg

    assert sg["AS/NZS3678:250"].get_f_y(0.008) == 280  # noqa: PLR2004
    assert sg["AS/NZS3678:250"].get_f_u(0.008) == 410  # noqa: PLR2004


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
