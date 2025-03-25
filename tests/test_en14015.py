"""
Tests for EN14015 class
"""

from utilityscripts.loads.en14015 import EN14015, SoilType


def test_en14015_init():
    """
    Basic test - can the tank be initialised?
    """

    h_l = 10.0
    h_t = 10.0
    r_e = 10.0
    t_shell = 10.0
    m_roof = 10.0
    rho_liquid = 10.0
    a_seismic = 10.0
    soil_profile = SoilType.A

    tank = EN14015(
        h_l=h_l,
        h_t=h_t,
        r_e=r_e,
        t_shell=t_shell,
        m_roof=m_roof,
        rho_liquid=rho_liquid,
        a_seismic=a_seismic,
        soil_profile=soil_profile,
    )

    assert tank.h_l == h_l
    assert tank.h_t == h_t
    assert tank.r_e == r_e
    assert tank.t_shell == t_shell
    assert tank.m_roof == m_roof
    assert tank.rho_liquid == rho_liquid
    assert tank.a_seismic == a_seismic
    assert tank.soil_profile == soil_profile

    # next test that the class can reach out into the data files.
    assert tank.x_1_ratio
    assert tank.x_2_ratio
    assert tank.t_1_ratio
    assert tank.t_2_ratio
    assert tank.k_s
