"""
Contains modules for working with concrete to AS3600
"""

from math import cos, pi, radians, sin, tan

D500N_STRESS_STRAIN = [[-0.05, -0.0025, 0, 0.0025, 0.05], [-500, -500, 0, 500, 500]]
D500L_STRESS_STRAIN = [[-0.015, -0.0025, 0, 0.0025, 0.015], [-500, -500, 0, 500, 500]]
R250N_STRESS_STRAIN = [[-0.05, -0.0025, 0, 0.0025, 0.05], [-500, -500, 0, 500, 500]]


def cot(angle) -> float:
    """
    Calculate the cotangent of an angle.

    Parameters
    ----------
    angle : float
        The angle to calculate the cotangent for.
        Should be in radians.

    Returns
    -------
    float
    """

    return 1 / tan(angle)


def alpha_2(f_c):
    """
    Calculate parameter alpha_2 as per AS3600-2018.

    :param f_c: The characteristic compressive strength of the concrete
    """

    return max(0.67, 0.85 - 0.0015 * f_c)


def gamma(f_c):
    """
    Calculate parameter gamma as per AS3600-2018.

    :param f_c: The characteristic compressive strength of the concrete.
    """

    return max(0.67, 0.97 - 0.0025 * f_c)


def generate_rectilinear_block(*, f_c, max_compression_strain: float = 0.003):
    """
    Generate a point on a rectilinear stress-strain curve as required by AS3600.

    :param f_c: the characteristic compressive strength of the concrete.
    :param max_compression_strain: the maximum allowable compressive strain in the
        concrete. According to AS3600 S8.1.3 this is 0.003.
    """

    gamma_val = gamma(f_c)
    min_strain = max_compression_strain * (1 - gamma_val)

    compressive_strength = alpha_2(f_c) * f_c

    return [
        [0, min_strain, max_compression_strain],
        [0, compressive_strength, compressive_strength],
    ]


def m_uo_min(*, z, f_ct_f, p_e=0, a_g=0, e=0):
    """
    Calculate the minimum required moment capacity.

    :param z: the uncracked section modulus, taken at the face of the section at which
        cracking occurs.
    :param f_ct_f: the characteristic flexural strength of the concrete.
    :param p_e: effective prestress force, accounting for losses.
    :param a_g: gross area.
    :param e: prestress eccentricity from the centroid of the uncracked section.
    """

    return 1.2 * (z * (f_ct_f + p_e / a_g) + p_e * e)


def m_uo():
    """
    Method to calculate the concrete capacity of a rectangular beam as per AS3600.
    """

    pass


def a_svmin_s8_2_1_7(*, f_c, f_sy, b_v, s: float = 1000):
    """
    Calculate the minimum transverse shear reinforcement required as per AS3600.

    This function computes the minimum area of transverse shear reinforcement required
    (if reinforcement is required).

    Parameters
    ----------
    f_c : float
        The characteristic compressive strength of concrete in MPa.
    f_sy : float
        The yield strength of the shear reinforcement in MPa.
    b_v : float
        The effective web width of the beam, in mm.
    s : float
        The spacing of the fitments in mm.
        If not provided the default value of 1000 mm is used.
        This gives an equivalent mm² / m value.

    Returns
    -------
    float
        The minimum area of transverse reinforcement (in mm²) required
        to ensure sufficient shear resistance.
    """

    return s * 0.08 * (f_c**0.5) * b_v / f_sy


def v_u_s8_2_3_1(
    *,
    f_c: float,
    b_v: float,
    d_v: float,
    k_v: float,
    a_sv: float,
    f_sy: float,
    theta_v: float,
    s: float = 1000,
    a_v: float = pi / 2,
):
    """
    Calculate the shear capacity of a beam.

    This consists of the concrete shear strength V_uc and the
    steel shear strength V_us.

    Notes
    -----
    The capacity may be limited by web crushing in shear.
    This is not accounted for in this function.

    Parameters
    ----------
    f_c : float
        The characteristic compressive strength of concrete in MPa.
    b_v : float
        The effective web width of the beam, in mm.
    d_v : float
        The effective shear depth of the beam, in mm.
    k_v : float
        The concrete shear strength modification factor.
    a_sv : float
        The shear area of a single (set of) fitment in mm².
    f_sy : float
        The yield strength of the shear reinforcement in MPa.
    theta_v: float
        The compressive strut angle in radians.
    s: float
        The spacing of shear fitments in mm.
        The default value is 1000mm.
        If A_sv is given in mm² / m then s does not need to be provided.
    a_v: float
        The angle of shear reinforcement to the longitudinal reinforcement, in radians.
        Default value is pi / 2 (90deg).

    Returns
    -------
    float
        The shear capacity in kN
    """

    return v_uc_s8_2_4_1(f_c=f_c, k_v=k_v, b_v=b_v, d_v=d_v) + v_us_s8_2_5_2(
        a_sv=a_sv, f_sy=f_sy, d_v=d_v, theta_v=theta_v, s=s, a_v=a_v
    )


def v_u_max_s8_2_3_2(
    *, f_c: float, b_v: float, d_v: float, theta_v: float, a_v: float = pi / 2
):
    """
    Calculate the maximum shear capacity of the beam as limited by web crushing.

    Parameters
    ----------
    f_c : float
        The characteristic compressive strength of concrete in MPa.
    b_v : float
        The effective web width of the beam, in mm.
    d_v : float
        The effective shear depth of the beam, in mm.
    theta_v : float
        The compressive strut angle, in radians.
    a_v : float
        The angle of shear reinforcement, in radians.
        Default is pi / 2 (90 degrees).

    Returns
    -------
    float
    The shear capacity in kN.
    """

    θv = theta_v

    part_a = 0.9 * f_c * b_v * d_v
    part_b = (cot(θv) + cot(a_v)) / (1 + cot(theta_v) ** 2)

    return 0.55 * (part_a * part_b) / 1000


def v_uc_s8_2_4_1(*, k_v, b_v, d_v, f_c):
    """
    Calculate the concrete contribution to shear capacity.
    As per AS3600 S8.2.4.1.

    Parameters
    ----------
    k_v : float
        The concrete shear strength modification factor.
        This depends on the shear strut angle, θ_v.
    b_v : float
        The width of the beam in mm.
    d_v : float
        The effective shear depth of the beam, in mm.
    f_c : float
        The characteristic compressive strength of concrete in MPa.

    Returns
    -------
    float
        The shear capacity in kN
    """

    return k_v * b_v * d_v * f_c**0.5 / 1000


def theta_v_s8_2_4_2(*, eta_x, return_radians: float = True):
    """
    Calculate the inclination of the concrete compressive strut.
    As per AS3600 S8.2.4.2.

    Parameters
    ----------
    eta_x : float
        The longitudinal strain in the concrete at the mid-height of the section, ε_x.

    Returns
    -------
    float
    The compressive strut angle.
    """

    theta_v = 29 + 7000 * eta_x

    if return_radians:
        return radians(theta_v)

    return theta_v


def k_v_s8_2_4_2(*, f_c, a_sv, a_svmin, eta_x, d_v, d_g: float = 20.0):
    """
    Calculate the concrete shear strength modification factor, k_v.
    As per AS3600 S8.2.4.2.

    Parameters
    ----------
    f_c : float
        The characteristic compressive strength of concrete in MPa.
    a_sv : float
        The shear reinforcement area provided, A_sv / s in mm².
        This should be normalised to apply over the same distance 's'
        as A_sv.min.
    a_svmin : float
        The minimum shear reinforcement area, A_sv.min / s in mm².
        This should be normalised to apply over the same distance 's'
        as A_sv.
    eta_x : float
        The longitudinal strain in the concrete at the mid-height of the section, ε_x.
    d_v : float
        The effective depth in shear in mm.
    d_g : float
        The nominal aggregate size, in mm.
        Default is 20mm.

    Returns
    -------
    float
    """

    k_v_base = 0.4 / (1 + 1500 * eta_x)

    if a_sv < a_svmin:
        k_dg = max(0.8, 32 / (16 + d_g)) if f_c < 65.0 else 2.0  # noqa: PLR2004

        return k_v_base * (1300 / (1000 + k_dg * d_v))

    return k_v_base


def eta_x_s8_2_4_2(
    *, mstar, vstar, nstar, d_v, e_s, a_st, e_c: float = 0.0, a_ct: float = 0.0
):
    """
    Calculate the longitudinal strain at the mid-height of the concrete section, ε_x.
    As per AS3600 S8.2.4.2.

    Notes
    -----
    The applied loads M* and V* should be absolute values.
    abs() is called on them before use regardless.

    The applied load N* is signed with tension +ve.
    This may be opposite what is used elsewhere in the code.

    If the initial calculation of eta_x is -ve,
    then a revised eta_x is calculated
    using the concrete elastic modulus and area e_c / a_ct.
    If either of these is 0.0 then eta_x is returned as 0.0.

    Parameters
    ----------
    mstar : float
        The applied bending moment M* in kNm.
        NOTE: this should be an absolute value.
        abs() is called on the value before calculation.
    vstar : float
        The applied shear V* in kN.
        NOTE: this should be an absolute value.
        abs() is called on the value before calculation.
    nstar : float
        The applied tensile load N* in kN.
        NOTE: +ve is tension, -ve is compression.
        This may be opposite to what is used elsewhere in the code.
    d_v : float
        The effective depth in shear in mm.
    e_s : float
        The elastic modulus of the tensile reinforcement in the tensile half of the beam.
    a_st : float
        The area of tensile reinforcement on the tension side of the beam
        (i.e. between D/2 and the extreme tensile fibre) in mm².
        If more tensile reinforcement is located between D/2 and the neutral axis it should be ignored.
        NOTE:
        if bars are not fully developed,
        their effective area should be reduced proportionately.
    e_c : float
        The elastic modulus of the concrete in the tensile half of the beam.
        NOTE: if e_c == 0 and eta_x is calculated as < 0  then 0.0 is returned.
    a_ct : float
        The area of concrete between the mid-depth
        (D/2) of the beam and the extreme tensile fibre, in mm².
        NOTE: if a_ct == 0 and eta_x is calculated as < 0  then 0.0 is returned.

    Returns
    -------
    float
    """

    max_eta_x = 3e-3
    min_eta_x = -2e-3

    # convert the loads to N, Nmm
    # because the bottom product E*A will result in N as units.

    mstar = abs(mstar) * 1e6  # convert to Nmm
    vstar = abs(vstar) * 1e3  # convert to N
    nstar = nstar * 1e3  # convert to N

    mstar = max(mstar, vstar * d_v)

    top = (mstar / d_v) + vstar + 0.5 * nstar
    bottom = 2 * e_s * a_st

    eta_x_base = min(top / bottom, max_eta_x)

    if eta_x_base > 0:
        return eta_x_base

    if e_c == 0 or a_ct == 0:
        return 0.0

    bottom = 2 * (e_s * a_st + e_c * a_ct)

    return min(max(top / bottom, min_eta_x), max_eta_x)


def theta_v_s8_2_4_3(*, return_radians: bool = True):
    """
    Determine the compressive strut angle for concrete.
    Uses the simplified method of AS3600 S8.2.4.3.

    Notes
    -----
    Only valid where f_c <65MPa and aggregate >= 10mm.

    Parameters
    ----------
    return_radians : bool
        Return the angle in radians or degrees?

    Returns
    -------
    float
    """

    theta_v = 36.0

    if return_radians:
        return radians(theta_v)

    return theta_v


def k_v_s8_2_4_3(*, a_sv, a_svmin, d_v):
    """
    Calculate the concrete shear strength modification factor, k_v.
    As per the simplified method of AS3600 S8.2.4.3.

    Notes
    -----
    Only valid where f_c <=65MPa and aggregate is >= 10mm.

    Parameters
    ----------
    a_sv : float
        The shear reinforcement area provided, A_sv / s in mm² or mm² / m.
        NOTE: should be normalised to apply over the same distance 's' as A_sv.min
    a_svmin : float
        The minimum shear reinforcement area, A_sv.min / s, in mm² or mm² / m.
        NOTE: should be normalised to apply over the same distance 's' as A_sv.
    d_v : float
        The effective depth in shear in mm.

    Returns
    -------
    float
    """

    k_v_max = 0.15

    if a_sv < a_svmin:
        return min(200 / (1000 + 1.3 * d_v), k_v_max)

    return k_v_max


def v_us_s8_2_5_2(*, a_sv, f_sy, d_v, theta_v, s: float = 1000, a_v: float = pi / 2):
    """
    Calculate the steel contribution to shear strength, V_us.
    As per AS3600 S8.2.5.2.

    Parameters
    ----------
    a_sv : float
        The shear area of a single (set of) fitment in mm^2
    f_sy : float
        The yield strength of the shear reinforcement.
    d_v : float
        The effective depth in shear in mm.
    theta_v : float
        The shear strut angle in radians
    s : float
        The spacing of shear fitments in mm.
        The default value is 1000 mm.
        If A_sv is given in mm^2 / m then s does not need to be provided.
    a_v : float
        The angle of the shear reinforcement to the longitudinal reinforcement,
        in radians.

    Returns
    -------
    float
        The steel contribution to the shear strength in kN.
    """

    steel_param = (a_sv * f_sy * d_v) / s
    angle_param = sin(a_v) * cot(theta_v) + cos(a_v)

    return steel_param * angle_param / 1000


def v_uo(*, f_c, u, d_om, beta_h=1.0):
    """
    Calculate the punching shear capacity.

    :param f_c: The characteristic strength of the concrete. Units are MPa.
    :param u: The punching shear perimeter. Units are m.
    :param d_om: The effective depth in punching shear. Units are m.
    :param beta_h: The shape factor for the punching shear area.
    :returns: Punching shear capacity in kN, provided input units are as specified.
    """

    def f_cv(f_c, beta_h=1.0):
        """
        Helper method to determine the punching shear strength.

        :param f_c: The characteristic strength of the concrete.
        :param beta_h: The shape factor for the punching shear area.
        """

        f_cv_1 = 0.17 * (1 + 2 / beta_h)
        f_cv_2 = 0.34

        return min(f_cv_1, f_cv_2) * f_c**0.5

    return f_cv(f_c, beta_h) * u * d_om * 1000  # *1000 to convert to kN.


def c_d(*, bar_spacing, cover, bar_type: str = "straight", narrow_element: bool = True):
    """
    Calculate the cover & spacing parameter c_d as per AS3600 S13.1.2

    :param bar_spacing: The spacing between bars.
    :param cover: The cover to the bars.
        See figure 13.1.2.2 to determine which cover is appropriate.
    :param bar_type: The type of bar, either "straight", "hooked" or "looped"
    :param narrow_element: Is the element narrow (e.g. beam web, column)
        or wide (e.g. slab, inner bars of a band beam etc.)
    """

    if bar_type not in ["straight", "hooked", "looped"]:
        raise ValueError("Incorrect bar_type.")

    if narrow_element:
        if bar_type in ["straight", "hooked"]:
            return min(bar_spacing / 2, cover)
        return cover

    if bar_type == "straight":
        return min(bar_spacing / 2, cover)
    if bar_type == "hooked":
        return bar_spacing / 2
    return cover


def k_3(
    *, d_b, bar_spacing, cover, bar_type: str = "straight", narrow_element: bool = True
):
    """
    Calculate parameter k_3 as per AS3600 S13.1.2.2

    :param d_b: bar diamter.
        Units should be consistent with bar_spacing and cover.
    :param bar_spacing: The spacing between bars.
    :param cover: The cover to the bars.
        See figure 13.1.2.2 to determine which cover is appropriate.
    :param bar_type: The type of bar, either "straight", "hooked" or "looped"
    :param narrow_element: Is the element narrow (e.g. beam web, column)
        or wide (e.g. slab, inner bars of a band beam etc.)
    """

    c_d_calc = c_d(
        bar_spacing=bar_spacing,
        cover=cover,
        bar_type=bar_type,
        narrow_element=narrow_element,
    )

    k_3_calc = 1 - 0.15 * (c_d_calc - d_b) / d_b

    return max(
        min(
            k_3_calc,
            1.0,
        ),
        0.7,
    )


def l_syt(*, f_c, f_sy, d_b, k_1=1.0, k_3=1.0, k_4=1.0, k_5=1.0):
    """
    Calculate the development length of a bar as per AS3600 S13.1.2.3

    :param f_c: The concrete characteristic compressive strength, f'c. In MPa.
    :param f_sy: The steel yield strength (in MPa)
    :param d_b: The diameter of the bar (in mm)
    :param k_1: The member depth parameter.
    :param k_3: The bar spacing parameter.
    :param k_4: Transverse reinforcement parameter.
    :param k_5: Transverse stress parameter.
    :return: The development length in mm.
    """

    k_2 = (132 - d_b) / 100
    k_prod = max(0.7, k_3 * k_4 * k_5)
    l_calc = 0.5 * k_1 * k_prod * f_sy * d_b / (k_2 * f_c**0.5)
    l_min = 0.058 * f_sy * k_1 * d_b

    return max(l_calc, l_min)
