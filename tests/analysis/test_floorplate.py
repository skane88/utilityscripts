from math import isclose

import pytest

from utilityscripts.analysis.floor_plate import (
    flat_plate_bending_point,
    flat_plate_bending_uniform,
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
