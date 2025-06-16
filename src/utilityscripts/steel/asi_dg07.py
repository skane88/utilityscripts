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
        The projected tearout area of a single bolt, in mm^2.
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
        The projected tearout area of the bolt group, in mm^2.
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
