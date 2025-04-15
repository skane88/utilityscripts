from math import isclose

import pytest

from utilityscripts.steel.as4100 import k_t, s9_2_2_4_v_bt, s9_2_2_4_v_by, s9_6_3_10_v_w


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


@pytest.mark.parametrize(
    "d_f, t_p, f_up, expected",
    [
        (0.020, 0.012, 410e6, 314880.0),
        (0.016, 0.020, 440e6, 450560.0),
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
    ],
)
def test_s9_2_2_4_v_bt(a_e, t_p, f_up, expected):
    """
    Test the s9_2_2_4_v_bt method.
    """

    assert isclose(s9_2_2_4_v_bt(a_e=a_e, t_p=t_p, f_up=f_up), expected, rel_tol=1e-6)


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
