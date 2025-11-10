"""
Python file to contain methods for working with AS5216.
"""

from enum import StrEnum


def phi_ms_tension(*, f_yf, f_uf):
    """
    Calculate the reduction factor for an anchor or channel in tension

    Parameters
    ----------
    f_yf :
        The yield strength of the fastener. In MPa.
    f_uf :
        The ultimate strength of the fastener. in MPa.
    """

    phi_base = 5 * f_yf / (6 * f_uf)

    return min(phi_base, 1 / 1.4)


def phi_ms_shear(*, f_yf, f_uf):
    """
    Calculate the reduction factor for an anchor or channel in shear

    Parameters
    ----------
    f_yf :
        The yield strength of the fastener. In MPa.
    f_uf :
        The ultimate strength of the fastener. in MPa.
    """

    f_ratio = f_yf / f_uf
    max_f_ratio = 0.8
    max_f_uf = 800

    if f_uf <= max_f_uf and f_ratio <= max_f_ratio:
        return min(f_ratio, 0.8)

    return 2 / 3


def phi_ms_ca():
    """
    The capacity reduction factor for an anchor-channel connection.
    """

    return 1 / 1.8


def phi_ms_l():
    """
    The capacity reduction factor for an anchor-channel lip in local failure.
    """

    return 1 / 1.8


def phi_ms_l_x(phi_inst):
    """
    The capacity reduction factor for an anchor-channel lip in local failure
    in the x-direction.

    Parameters
    ----------
    phi_inst :
        A reduction factor for installation issues as per Appendix A.
    """

    return phi_ms_l() * phi_inst


def phi_ms_flex():
    """
    The capacity reduction factor for flexural failure of a cast in channel.
    """

    return 1 / 1.15


def phi_ms_re():
    """
    The capacity reduction factor for supplementary reinforcement.
    """

    return 0.8


def phi_mc(phi_inst):
    """
    The capacity reduction factor for concrete.

    Parameters
    ----------
    phi_inst :
        A reduction factor for installation issues as per T3.2.4 or Appendix A
    """

    phi_c = 1 / 1.5

    return phi_c * phi_inst


def phi_m_sp(phi_inst):
    """
    The capacity reduction factor for concrete splitting.

    Parameters
    ----------
    phi_inst :
        A reduction factor for installation issues as per T3.2.4 or Appendix A
    """

    return phi_mc(phi_inst=phi_inst)


def phi_m_p(phi_inst):
    """
    The capacity reduction factor for concrete pull out.

    Parameters
    ----------
    phi_inst :
        A reduction factor for installation issues as per T3.2.4 or Appendix A
    """

    return phi_mc(phi_inst=phi_inst)


class Concrete(StrEnum):
    CRACKED = "cracked"
    UNCRACKED = "uncracked"


def cracked(*, sigma_l, sigma_r: float = 3.0, f_ct: float = 0.0):
    """
    Is the concrete considered to be cracked or uncracked?

    Notes
    -----
    - For 2D elements (slabs etc) this needs to be checked in both directions.

    Parameters
    ----------
    sigma_l : float
        The stress in the concrete due to loads. In MPa.
    sigma_r : float
        The stress in the concrete due to restraint, shrinkage etc. In MPa.
        Recommended to be at least 3.0MPa
    f_ct : float
        The characteristic tensile strength of the concrete.
        Can conservatively be taken to be 0MPa
    """

    sigma_total = sigma_l + sigma_r

    return Concrete.UNCRACKED if sigma_total <= f_ct else Concrete.CRACKED


def s6_2_3_1_n_rkc(*, n_0_rkc, a_cn, a_0_cn, psi_s_n, psi_re_n, psi_ec_n, psi_m_n):
    """
    Calculate the concrete cone failure capacity.

    Parameters
    ----------
    n_0_rkc :
        The reference characteristic strength.
    a_cn :
        The projected cone failure area.
    a_0_cn :
        The reference projected cone failure arwa
    psi_s_n :
        The edge parameter to account for the effects of edges.
    psi_re_n :
        The shell spalling parameter.
    psi_ec_n :
        The eccentricity parameter.
    psi_m_n :
        The axial force parameter.
    """

    return n_0_rkc * (a_cn / a_0_cn) * psi_s_n * psi_re_n * psi_ec_n * psi_m_n


def s6_2_3_2_n_0_rkc(*, k_1, f_c, h_ef):
    """
    The concrete cone failure capacity of a single anchor in reference conditions.

    Parameters
    ----------
    k_1 :
        The tensile loading factor. This varies based on anchor type and concrete type.
        See Appendix A or s6.2.3.2.
    f_c :
        The concrete characteristic strength. In MPa.
    h_ef :
        Anchor embedment length.
    """

    return k_1 * f_c**0.5 * h_ef**1.5


def s6_2_3_3_c_cr_n(*, h_ef):
    """
    Characteristic edge distance of a reference anchor

    Parameters
    ----------
    h_ef :
        Embedment depth.
    """

    return 1.5 * h_ef


def s6_2_3_3_s_cr_n(*, h_ef):
    """
    Characteristic spacing required to allow development of the full anchor capacity.

    Parameters
    ----------
    h_ef :
        Embedment depth.
    """

    return 2 * s6_2_3_3_c_cr_n(h_ef=h_ef)


def s6_2_3_3_a_0_cn(*, h_ef):
    """
    Reference projected area of a reference anchor

    Parameters
    ----------
    h_ef :
        Embedment depth.
    """

    return s6_2_3_3_s_cr_n(h_ef=h_ef) ** 2


def s6_2_3_4_psi_s_n(*, c, c_cr_n):
    """
    Edge effect factor for the effects of an edge.

    Parameters
    ----------
    c :
        The actual edge distance.
    c_cr_n :
        The characteristic edge distance of a reference anchor.
    """

    return min(0.7 + 0.3 * (c / c_cr_n), 1.0)


def s6_2_3_5_psi_re_n(*, h_ef):
    """
    The shell spalling factor. This accounts for a potential plane of weakness
    introduced by layers of reinforcement.

    Notes
    -----
    - AS5216 includes requirements on reinforcement size and spacing.
      If these are met this factor may be taken as 1.0.

    Parameters
    ----------
    h_ef :
        The embedment depth. In mm.
    """

    return min(0.5 + h_ef / 200, 1.0)


def s6_2_3_6_psi_ec_n(*, e_n, s_cr_n):
    """
    The eccentricity factor. This accounts for the effects of eccentricity which
    result in uneven loads on anchors.

    Parameters
    ----------
    e_n :
        Eccentricity of the resultant tensile force on the anchor pattern.
    s_cr_n :
        The minimum spacing required to allow an anchor to achieve its
        reference capacity.
    """

    return min(1.0, 1 / (1 + 2 * e_n / s_cr_n))
