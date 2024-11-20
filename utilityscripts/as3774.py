"""
Functions for working with material loads in bins to AS3774.
"""

from math import cos, exp, radians, sin


class Material:
    def __init__(
        self,
        *,
        unit_weight,
        phi_w: tuple[float, float] | float,
        phi_i: tuple[float, float] | float,
        phi_r: float,
        use_radians: bool = True,
        material_name: str | None = None,
    ):
        """
        Initialize a new Material instance.

        Parameters
        ----------
        unit_weight : float
            The unit weight of the material (kN/m³).
        phi_w : tuple[float, float] | float
            The angle of wall friction (radians).
            Provide as a tuple with the min & max values.
            If a single float is provided it is used for both
            the min & max values.
        phi_i : tuple[float, float] | float
            The internal friction angle (radians).
            Provide as a tuple with the min & max values.
            If a single float is provided it is used for both
            the min & max values.
        phi_r : float
            The angle of repose (in radians)
        use_radians : bool
            Are angles in radians or not?
        material_name : str | None
            The name of the material.
            Provided for convenience.
        """

        self._material_name = material_name
        self._unit_weight = unit_weight

        if isinstance(phi_w, float):
            phi_w = (phi_w, phi_w)

        if isinstance(phi_i, float):
            phi_i = (phi_i, phi_i)

        if not use_radians:
            phi_w = (radians(phi_w[0]), radians(phi_w[1]))
            phi_i = (radians(phi_i[0]), radians(phi_i[1]))
            phi_r = radians(phi_r)

        self._phi_w = phi_w
        self._phi_i = phi_i
        self._phi_r = phi_r

    @property
    def material_name(self):
        """
        The name of the material.

        Returns
        -------
        str
        """
        return self._material_name

    @property
    def unit_weight(self):
        """
        The unit weight of the material.

        Returns
        -------
        float
        """
        return self._unit_weight

    @property
    def phi_w(self):
        """
        The angle of wall friction.

        Returns
        -------
        tuple[float, float]
        """
        return self._phi_w

    @property
    def phi_w_min(self):
        """
        The minimum wall friction angle.

        Returns
        -------
        float
        """
        return min(self.phi_w)

    @property
    def phi_w_max(self):
        """
        The maximum wall friction angle.

        Returns
        -------
        float
        """
        return max(self.phi_w)

    @property
    def phi_i(self):
        """
        The angle of internal friction.

        Returns
        -------
        tuple[float, float]
        """
        return self._phi_i

    @property
    def phi_i_min(self):
        """
        The minimum internal friction angle.

        Returns
        -------
        float
        """
        return min(self.phi_i)

    @property
    def phi_i_max(self):
        """
        The maximum internal friction angle.

        Returns
        -------
        float
        """
        return max(self.phi_i)

    @property
    def phi_r(self):
        """
        The angle of repose.

        Returns
        -------
        float
        """
        return self._phi_r

    def __repr__(self):
        return (
            f"Material(unit_weight={self._unit_weight:.2f}, "
            f"phi_w=({', '.join(f'{num:.3f}' for num in self._phi_w)}), "
            f"phi_i=({', '.join(f'{num:.3f}' for num in self._phi_i)}), "
            f"phi_r={self._phi_r:.3f}, "
            f"material_name={self._material_name!r})"
        )


class Hopper:
    def __init__(self, material: Material, hopper_name: str | None = None):
        """
        Initialize a Hopper object to representate a bulk solids storage container.

        Parameters
        ----------
        material : Material
            A Material object to represent the stored material.
        hopper_name : str
            An optional name for the hopper.
        """

        self._hopper_name = hopper_name
        self._material = material


def p_ni(gamma: float, r_c: float, c_z: float, mu: float):
    """
    Calculate the initial normal pressure on vertical walls (p_ni) in accordance with AS3774.

    Parameters
    ----------
    gamma : float
        The unit weight of the material (kN/m³).
    r_c : float
        The characteristic radius of the container.
    c_z : float
        The Janssen depth function (dimensionless).
    mu : float
        The coefficient of wall friction (tan(phi_w)).

    Returns
    -------
    float
    """

    return gamma * r_c * c_z / mu


def c_z(z, z_o):
    """
    Calculates the Janssen depth function.

    Parameters
    ----------
    z : float
        The depth at which the value is calculated.
    z_o : float
        The characteristic depth of the hopper.

    Returns
    -------
    float
    """

    return 1 - exp(-z / z_o)


def z_o(r_c, mu, k_lateral):
    """
    Calculate the characteristic depth of the container.

    Parameters
    ----------
    r_c : float
        The characteristic radius of the container.
    mu : float
        The coefficient of wall friction (tan(phi_w)).
    k_lateral : float
        The lateral pressure coefficient.

    Returns
    -------
    float
    """
    return r_c / (mu * k_lateral)


def k_lateral(phi_i, mu):
    """
    Calculate the lateral pressure coefficient.

    Parameters
    ----------
    phi_i : float
        Internal friction angle in radians.
    mu : float
        Coefficient of wall friction (tan(phi_w)).

    Returns
    -------
    float
    """

    k_calc = (
        1 + sin(phi_i) ** 2 - 2 * (sin(phi_i) ** 2 - mu**2 * cos(phi_i) ** 2) ** 0.5
    ) / (4 * mu**2 + cos(phi_i) ** 2)

    return max(k_calc, 0.35)


def p_ni_6_2_1_2(z, z_o, h_o, gamma, r_c, mu):
    """
    Computes modified initial normal pressures for squat containers.

    This function calculates the pressure p_ni for normal pressures in squat containers
    as modified by AS3774 S6.2.1.2.
    This only applies above z = 1.5 * h_o.

    Parameters
    ----------
    z : float
        The depth at which the value is calculated.
    z_o : float
        The characteristic depth of the hopper.
    h_o : float
        Height of the reference surface above the wall contact.
    gamma : float
        The density of the material.
    r_c : float
        The characteristic radius of the container.
    mu : float
        The coefficient of wall friction (tan(phi_w)).

    Returns
    -------
    float
    """

    if z < h_o:
        return 0

    if z < 1.5 * h_o:
        c_1 = c_z(z=1.5 * h_o, z_o=z_o)
        p_1 = p_ni(gamma=gamma, r_c=r_c, c_z=c_1, mu=mu)

        return ((z - h_o) / (0.5 * h_o)) * p_1

    return p_ni(gamma=gamma, r_c=r_c, c_z=c_z(z=z, z_o=z_o), mu=mu)


def psi(e_s, e_w, d_c, t):
    """
    Calculate the flexibility reduction factor for very flexible containers,
    as per AS3774 S6.2.1.8.

    Parameters
    ----------
    e_s : float
        The elastic modulus of the stored material.
    e_w : float
        The elastic modulus of the wall.
    d_c : float
        The diameter of the container (cylinders)
        or the largest inscribed diameter for other shapes.
    t : float
        The wall thickness.

    Returns
    -------
    float
    """

    return max(0.85, 1 - (e_s * d_c / 2) / (e_w * t))
