"""
Test the AS4100 methods.
"""

from math import isclose

import pytest

from utilityscripts.steel.as4100 import (
    RestraintCode,
    WebType,
    s5_6_3_k_t,
    s5_11_5_alpha_v,
    s9_1_9_block,
    s9_2_2_4_v_bt,
    s9_2_2_4_v_by,
    s9_4_1_v_f,
    s9_4_2_v_b,
    s9_4_3_m_p,
    s9_6_3_10_v_w,
)
from utilityscripts.steel.design import ISection
from utilityscripts.steel.steel import standard_grades


def test_i_section():
    steel = standard_grades()["UB"]

    assert ISection(
        section_name="test", steel=steel, b_f=0.15, d=0.36, t_f=0.012, t_w=0.008
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
        ((RestraintCode.P, RestraintCode.P), 0.2060, 4.00, 0.0120, 0.0065, 1, 1.081012),
        ((RestraintCode.P, RestraintCode.F), 0.2834, 10.0, 0.0118, 0.0067, 1, 1.019352),
        ((RestraintCode.F, RestraintCode.F), 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
        ((RestraintCode.F, RestraintCode.L), 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
        ((RestraintCode.L, RestraintCode.F), 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
        ((RestraintCode.L, RestraintCode.L), 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
        ((RestraintCode.F, RestraintCode.U), 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
        ((RestraintCode.U, RestraintCode.F), 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
    ],
)
def test_k_t(restraint_code, d_1, length, t_f, t_w, n_w, expected):
    """
    Test the k_t method.
    """

    k_t_calc = s5_6_3_k_t(
        restraint_code=restraint_code, d_1=d_1, length=length, t_f=t_f, t_w=t_w, n_w=n_w
    )

    assert isclose(k_t_calc, expected, rel_tol=1e-6)


@pytest.mark.parametrize(
    "d_p, t_w, f_y, s, f_y_ref, web_type, expected",
    [
        (1.000, 0.008, 300e6, 0.800, 250e6, WebType.STIFFENED, 0.918947),
    ],
)
def test_s5_11_5_alpha_v(d_p, t_w, f_y, s, f_y_ref, web_type, expected):
    """
    Test the s5_11_5_alpha_v method.
    """

    # TODO: add more tests

    assert isclose(
        s5_11_5_alpha_v(
            d_p=d_p, t_w=t_w, f_y=f_y, s=s, f_y_ref=f_y_ref, web_type=web_type
        ),
        expected,
        rel_tol=1e-6,
    )


@pytest.mark.parametrize(
    "d_f, t_p, f_up, expected",
    [
        (0.020, 0.012, 410e6, 314880.0),
        (0.016, 0.020, 440e6, 450560.0),
        (0.020, 0.020, 440e6, 563200.0),  # Steel Designer's handbook 8ed, p.262
        (0.020, 0.008, 440e6, 225280.0),  # Steel Designer's handbook 8ed, p.262
    ],
)
def test_s9_2_2_4_v_by(d_f, t_p, f_up, expected):
    """
    Test the s9_2_2_4_v_by method.
    """

    assert isclose(s9_2_2_4_v_by(d_f=d_f, t_p=t_p, f_up=f_up), expected, rel_tol=1e-6)


@pytest.mark.parametrize(
    "a_e, t_p, f_up, expected",
    [
        (0.035, 0.012, 410e6, 172200.0),
        (0.028, 0.020, 440e6, 246400.0),
        (0.035, 0.020, 440e6, 308000.0),  # Steel Designer's handbook 8ed, p.262
        (0.048, 0.008, 440e6, 168960.0),  # Steel Designer's handbook 8ed, p.262
    ],
)
def test_s9_2_2_4_v_bt(a_e, t_p, f_up, expected):
    """
    Test the s9_2_2_4_v_bt method.
    """

    assert isclose(s9_2_2_4_v_bt(a_e=a_e, t_p=t_p, f_up=f_up), expected, rel_tol=1e-6)


@pytest.mark.parametrize(
    "a_gv, a_nv, a_nt, f_yp, f_up, k_bs, expected",
    [
        (
            4200e-6,
            2880e-6,
            1360e-6,
            280e6,
            440e6,
            1.0,
            1304000.0,
        ),  # Steel Designer's Handbook, 8ed, Example E.2, P441.
        (
            2100e-6,
            1440e-6,
            2240e-6,
            280e6,
            440e6,
            1.0,
            1338400.0,
        ),  # Steel Designer's Handbook, 8ed, Example E.2, P442.
    ],
)
def test_s9_1_9_block(a_gv, a_nv, a_nt, f_yp, f_up, k_bs, expected):
    """
    Test the s9_1_9_block method.
    """

    assert isclose(
        s9_1_9_block(a_gv=a_gv, a_nv=a_nv, a_nt=a_nt, f_yp=f_yp, f_up=f_up, k_bs=k_bs),
        expected,
        rel_tol=1e-6,
    )


@pytest.mark.parametrize(
    "d_pin, n_s, f_yp, expected",
    [
        (0.016, 1, 250e6, 31160.0),
        (0.020, 2, 300e6, 116900.0),
    ],
)
def test_s9_4_1_v_f(d_pin, n_s, f_yp, expected):
    """
    Test the s9_4_1_v_f method.
    """

    assert isclose(s9_4_1_v_f(d_pin=d_pin, n_s=n_s, f_yp=f_yp), expected, rel_tol=1e-2)


@pytest.mark.parametrize(
    "d_pin, t_p, f_yp, k_p, expected",
    [
        (0.016, 0.012, 250e6, 1.0, 67200.0),
        (0.020, 0.020, 300e6, 0.5, 84000.0),
    ],
)
def test_s9_4_2_v_b(d_pin, t_p, f_yp, k_p, expected):
    """
    Test the s9_4_2_v_b method.
    """

    assert isclose(
        s9_4_2_v_b(d_pin=d_pin, t_p=t_p, f_yp=f_yp, k_p=k_p), expected, rel_tol=1e-2
    )


@pytest.mark.parametrize(
    "d_pin, f_yp, expected",
    [
        (0.016, 250e6, 171.0),
        (0.020, 300e6, 400.0),
    ],
)
def test_s9_4_3_m_p(d_pin, f_yp, expected):
    """
    Test the s9_4_3_m_p method.
    """

    assert isclose(s9_4_3_m_p(d_pin=d_pin, f_yp=f_yp), expected, rel_tol=1e-2)


@pytest.mark.parametrize(
    "size, f_uw, k_r, phi_weld, expected",
    [
        (0.006, 490e6, 1.0, 0.8, 998000.0),
        (0.008, 490e6, 1.0, 0.8, 1330000.0),
    ],
)
def test_s9_6_3_10_v_w(size, f_uw, k_r, phi_weld, expected):
    """
    Test the s9_6_3_10_v_w method.
    """

    assert isclose(
        s9_6_3_10_v_w(size=size, f_uw=f_uw, k_r=k_r, phi_weld=phi_weld),
        expected,
        rel_tol=1e-2,
    )
