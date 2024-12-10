"""
Contains modules for working with concrete to AS3600
"""

D500N_STRESS_STRAIN = [[-0.05, -0.0025, 0, 0.0025, 0.05], [-500, -500, 0, 500, 500]]
D500L_STRESS_STRAIN = [[-0.015, -0.0025, 0, 0.0025, 0.015], [-500, -500, 0, 500, 500]]
R250N_STRESS_STRAIN = [[-0.05, -0.0025, 0, 0.0025, 0.05], [-500, -500, 0, 500, 500]]


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
