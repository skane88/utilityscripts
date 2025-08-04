"""
Test functions in AS3600
"""

from math import isclose

import pytest

from utilityscripts.concrete.as3600 import Concrete, S816SectType, s8_1_6_1_a_st_min


def test_concrete_init():
    concrete = Concrete(f_c=20)
    assert concrete.f_c == 20  # noqa: PLR2004
    assert concrete.density == 2400  # noqa: PLR2004
    assert concrete.epsilon_c == 0.003  # noqa: PLR2004
    assert concrete.poisson_ratio == 0.2  # noqa: PLR2004
    assert isclose(concrete.f_cm, 25, rel_tol=1e-3)
    assert isclose(concrete.f_cmi, 22, rel_tol=1e-3)
    assert isclose(concrete.f_ctf, 2.683, rel_tol=1e-3)
    assert isclose(concrete.f_ct, 1.61, rel_tol=1e-3)
    assert isclose(concrete.elastic_modulus, 23700, rel_tol=1e-3)
    assert isclose(concrete.alpha_2, 0.82, rel_tol=1e-3)
    assert isclose(concrete.gamma, 0.92, rel_tol=1e-3)

    concrete = Concrete(20)
    assert concrete.f_c == 20  # noqa: PLR2004
    assert concrete.density == 2400  # noqa: PLR2004
    assert concrete.epsilon_c == 0.003  # noqa: PLR2004
    assert concrete.poisson_ratio == 0.2  # noqa: PLR2004
    assert isclose(concrete.f_cm, 25, rel_tol=1e-3)
    assert isclose(concrete.f_cmi, 22, rel_tol=1e-3)
    assert isclose(concrete.f_ctf, 2.683, rel_tol=1e-3)
    assert isclose(concrete.f_ct, 1.61, rel_tol=1e-3)
    assert isclose(concrete.elastic_modulus, 23700, rel_tol=1e-3)
    assert isclose(concrete.alpha_2, 0.82, rel_tol=1e-3)
    assert isclose(concrete.gamma, 0.92, rel_tol=1e-3)


@pytest.mark.parametrize("f_c", [20, 25, 32, 40, 50, 65])
def test_f_ctf_s3_1_1_3(f_c):
    assert isclose(Concrete(f_c=f_c).f_ctf, (0.6 * (f_c**0.5)), rel_tol=1e-6)


@pytest.mark.parametrize("f_c", [20, 25, 32, 40, 50, 65])
def test_f_ct_s3_1_1_3(f_c):
    assert isclose(Concrete(f_c=f_c).f_ct, (0.36 * (f_c**0.5)), rel_tol=1e-6)


@pytest.mark.parametrize(
    "f_c, expected", [(20, 25), (25, 31), (32, 39), (40, 48), (50, 59), (65, 75)]
)
def test_f_cm(f_c, expected):
    assert isclose(Concrete(f_c=f_c).f_cm, expected, rel_tol=1e-6)


@pytest.mark.parametrize(
    "f_cmi, expected", [(20, 22), (25, 28), (32, 35), (40, 43), (50, 53), (65, 68)]
)
def test_f_cmi(f_cmi, expected):
    assert isclose(Concrete(f_c=f_cmi).f_cmi, expected, rel_tol=1e-6)


@pytest.mark.parametrize(
    "d_beam, d_reo, b_w, f_ct_f, f_sy, s_type, expected",
    [(600, 534, 750, 3.39, 500, S816SectType.RECTANGULAR, 686)],
)
def test_s8_1_6_1_a_st_min(
    d_beam, d_reo, b_w, f_ct_f, f_sy, s_type: S816SectType, expected
):
    assert isclose(
        s8_1_6_1_a_st_min(
            d_beam=d_beam,
            d_reo=d_reo,
            b_w=b_w,
            f_ct_f=f_ct_f,
            f_sy=f_sy,
            sect_type=s_type,
        ),
        expected,
        rel_tol=1e-3,
    )
