"""
Checks related to ASI DG07 - pinned baseplates
"""


def a_no(*, h_ef: float) -> float:
    """
    Calculates the projected tearout area of a single bolt.

    Parameters
    ----------
    h_ef : float
        The effective embedded length of the bolt, in mm. Typ. the distance to the top
        of the washer plate or the top of the bolt head (if no washer plate).

    Returns
    -------
    float
        The projected tearout area of a single bolt, in mm².
    """

    return 9 * h_ef**2


def a_n_group(*, h_ef: float, s_p: float, s_g: float, t_c: float) -> float:
    """
    Calculates the projected tearout area of a group of bolts.

    Parameters
    ----------
    h_ef : float
        The effective embedded length of the bolts, in mm. Typ. the distance to the top
        of the washer plate or the top of the bolt head (if no washer plate).
    s_p : float
        The pitch of the bolts in the x-direction. In mm.
    s_g : float
        The pitch of the bolts in the y-direction. In mm.
        t_c : float
        The thickness of the concrete, in mm.

    Returns
    -------
    float
        The projected tearout area of the bolt group, in mm².
    """

    x_0 = max(s_p - 3 * (t_c - h_ef), 0.0)
    y_0 = max(s_g - 3 * (t_c - h_ef), 0.0)

    x_1 = 3 * h_ef + s_p
    y_1 = 3 * h_ef + s_g

    return x_1 * y_1 - x_0 * y_0


def phi_n_c2(
    *,
    n_b: int,
    h_ef: float,
    s_p: float,
    s_g: float,
    t_c: float,
    f_c: float,
    phi_po: float = 0.70,
) -> float:
    """
    Calculates the design strength of a group of bolts.

    Parameters
    ----------
    n_b : int
        The number of bolts in the group. Should be either 1 or 4.
    h_ef : float
        The effective embedded length of the bolts, in mm. Typ. the distance to the top
        of the washer plate or the top of the bolt head (if no washer plate).
    s_p : float
        The pitch of the bolts in the x-direction. In mm.
    s_g : float
        The pitch of the bolts in the y-direction. In mm.
    t_c : float
        The thickness of the concrete, in mm.
    f_c : float
        The characteristic compressive strength of the concrete, in MPa.
    """

    if n_b != 1 and n_b != 4:  # noqa: PLR2004, PLR1714
        raise ValueError("n_b must be either 1 or 4")

    if h_ef > 635.0:  # noqa: PLR2004
        raise ValueError("h_ef must be less than 635.0 mm")

    a_no_calc = a_no(h_ef=h_ef)

    a_n = a_n_group(h_ef=h_ef, s_p=s_p, s_g=s_g, t_c=t_c) if n_b == 4 else a_no_calc  # noqa: PLR2004

    alpha = 10.1 if h_ef < 280.0 else 3.90  # noqa: PLR2004
    beta = 1.5 if h_ef < 280.0 else 1.67  # noqa: PLR2004

    return phi_po * alpha * (f_c**0.5) * (h_ef**beta) * (a_n / a_no_calc) * 1e-3


def chk08_y2_pfc(*, b_fc1: float, d_h: float) -> float:
    """
    Calculate the yield line parameter y_2 for a PFC.

    Parameters
    ----------
    b_fc1 : float
        The clear flange width of the PFC
    d_h : float
        The hole diameter in the baseplate.
    """

    return ((2 * b_fc1 - d_h) * b_fc1) ** 0.5


def chk08_alpha_pfc(
    *, b_fc1: float, d_h: float, x_2: float, a_b: float, s_p: float
) -> float:
    """
    Calculate the yield line parameter alpha for a PFC.

    Parameters
    ----------
    b_fc1 : float
        The clear flange width of the PFC
    d_h : float
        The hole diameter in the baseplate.
    x_2 : float
        The distance from the face of the PFC web to the centreline of the bolthole
    a_b : float
        The distance from the bolthole to the inside face of the flange
    s_p : float
        The pitch of the bolts.
    """

    y_2 = chk08_y2_pfc(b_fc1=b_fc1, d_h=d_h)
    y_c = min(a_b, y_2)

    alpha_a = (2 * b_fc1**2 - b_fc1 * d_h + y_2**2) / (y_2 * x_2)
    alpha_b = (b_fc1 * (2 * b_fc1 - d_h) * (a_b + y_2) + (y_2 + a_b) * a_b * y_2) / (
        2 * a_b * y_2 * x_2
    )

    if y_2 < s_p / 2 and y_2 > a_b:
        return alpha_b

    if y_2 < s_p / 2:
        return min(alpha_a, alpha_b)

    y_d = min(a_b, ((2 * b_fc1 - d_h) / 2) ** 0.5)

    alpha_c = (4 * b_fc1**2 - 2 * d_h * b_fc1 + 2 * y_c**2 + s_p * y_c) / (
        4 * y_c * x_2
    )
    alpha_d = (2 * b_fc1 * x_2 - d_h * x_2 + 2 * y_d**2 + s_p * y_d - d_h * y_d) / (
        2 * y_d * x_2
    )
    alpha_e = (b_fc1 * x_2 - d_h * x_2 + 2 * a_b**2 + a_b * s_p - a_b * d_h) / (
        2 * a_b * x_2
    )

    return min(alpha_c, alpha_d, alpha_e)
