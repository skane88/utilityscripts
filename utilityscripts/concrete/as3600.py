"""
Contains modules for working with concrete to AS3600
"""

import sys
from enum import StrEnum
from math import cos, pi, radians, sin, tan

import numpy as np
from matplotlib import pyplot as plt

D500N_STRESS_STRAIN = [[-0.05, -0.0025, 0, 0.0025, 0.05], [-500, -500, 0, 500, 500]]
D500L_STRESS_STRAIN = [[-0.015, -0.0025, 0, 0.0025, 0.015], [-500, -500, 0, 500, 500]]
R250N_STRESS_STRAIN = [[-0.05, -0.0025, 0, 0.0025, 0.05], [-500, -500, 0, 500, 500]]


STANDARD_GRADES = {
    "f_c": [20, 25, 32, 40, 50, 65, 80, 100, 120],
    "f_cm": [25, 31, 39, 48, 59, 75, 91, 110, 128],
    "f_cmi": [22, 28, 35, 43, 53, 68, 82, 99, 115],
    "E_c": [24000, 26700, 30100, 32800, 34800, 37400, 39600, 42200, 44400],
}


class Ductility(StrEnum):
    """
    An Enum to represent the ductility grades of steel.
    """

    N = "N"
    L = "L"


class SectType813(StrEnum):
    """
    An Enum to represent the 3x types of section referred to in S8.1.3 note 2.
    """

    RECTANGULAR = "rectangular"
    CIRCULAR = "circular"
    OTHER = "other"


def get_f_cm(f_c):
    """
    Determine the mean compressive strength, based on the values given in
    AS3600-2018 Table 3.1.2
    """

    f_c_vals = np.asarray(STANDARD_GRADES["f_c"])
    f_cm_vals = np.asarray(STANDARD_GRADES["f_cm"])

    return np.interp(f_c, f_c_vals, f_cm_vals)


def get_f_cmi(f_c):
    """
    Determine the mean in-situ compressive strength, based on the values given in
    AS3600-2018 Table 3.1.2
    """

    f_c_vals = np.asarray(STANDARD_GRADES["f_c"])
    f_cmi_vals = np.asarray(STANDARD_GRADES["f_cmi"])

    return np.interp(f_c, f_c_vals, f_cmi_vals)


class Concrete:
    """
    Class to represent a concrete material.
    """

    def __init__(
        self,
        f_c: float,
        *,
        density: float = 2400,
        eta_c: float = 0.003,
        f_cm: float | None = None,
        f_cmi: float | None = None,
        elastic_modulus: float | None = None,
        sect_type: SectType813 = SectType813.RECTANGULAR,
    ):
        """
        Initialise the concrete material.

        Parameters
        ----------
        f_c : float
            The characteristic compressive strength of the concrete.
        density : float
            The density of the concrete.
            Default value is 2400 kg/m³.
        eta_c : float
            The maximum compressive strain of the concrete.
            Default value is 0.003.
        f_cm : float
            The mean compressive strength of the concrete.
            If not provided, it is calculated based on the characteristic
            compressive strength.
        f_cmi : float
            The mean in-situ compressive strength of the concrete.
            If not provided, it is calculated based on the characteristic
            compressive strength.
        elastic_modulus : float
            The elastic modulus of the concrete.
            If not provided, it is calculated based on the characteristic
            compressive strength.
        sect_type : SectType813
            The type of section the concrete is formed into.
            This is used to reduce alpha_2 as per note 2 in S8.1.3.
        """

        self._f_c = f_c
        self._density = density
        self._eta_c = eta_c
        self._f_cm = f_cm
        self._f_cmi = f_cmi
        self._elastic_modulus = elastic_modulus
        self._sect_type = sect_type

    @property
    def f_c(self):
        """
        Return the characteristic compressive strength of the concrete.
        """

        return self._f_c

    @property
    def density(self):
        """
        Return the density of the concrete.
        """

        return self._density

    @property
    def eta_c(self):
        """
        Return the maximum compressive strain of the concrete.
        """

        return self._eta_c

    @property
    def f_cm(self):
        """
        Return the mean compressive strength of the concrete.

        Notes
        -----
        If the value was not provided, it is calculated based on Table 3.1.2 in
        AS3600-2018.
        """

        if self._f_cm is None:
            self._f_cm = get_f_cm(self._f_c)

        return self._f_cm

    @property
    def f_cmi(self):
        """
        Return the mean in-situ compressive strength of the concrete.

        Notes
        -----
        If the value was not provided, it is calculated based on Table 3.1.2 in
        AS3600-2018.
        """

        if self._f_cmi is None:
            self._f_cmi = get_f_cmi(self._f_c)

        return self._f_cmi

    @property
    def elastic_modulus(self):
        """
        Return the elastic modulus of the concrete.

        Notes
        -----
        If the value was not provided, it is calculated based on Table 3.1.2 in
        AS3600-2018.
        """

        if self._elastic_modulus is not None:
            return self._elastic_modulus

        if self.f_cmi <= 40:  # noqa: PLR2004
            self._elastic_modulus = (self.density**1.5) * (0.043 * (self.f_cmi) ** 0.5)
        else:
            self._elastic_modulus = (self.density**1.5) * (
                0.024 * (self.f_cmi) ** 0.5 + 0.12
            )

        return self._elastic_modulus

    @property
    def f_ctf(self):
        """
        Calculate the characteristic flexural strength of the concrete
        as per AS3600 S3.1.1.3.
        """

        return 0.6 * (self.f_c**0.5)

    @property
    def f_ct(self):
        """
        Calculate the characteristic tensile strength of the concrete
        as per AS3600 S3.1.1.3.
        """

        return 0.36 * (self.f_c**0.5)

    @property
    def alpha_2(self):
        """
        Calculate parameter alpha_2 as per AS3600-2018.
        """

        return alpha_2(f_c=self.f_c, sect_type=self._sect_type)

    @property
    def gamma(self):
        """
        Calculate parameter gamma as per AS3600-2018.
        """

        return gamma(self.f_c)

    @property
    def rect_strain(self):
        """
        Return the minimum strain for the rectangular stress block
        """

        return self.max_comp_strain * (1 - self.gamma)

    @property
    def _strain_values(self):
        """
        Return the strain values for the stress-strain curve.
        """

        return np.asarray(
            [
                0.0,
                self.rect_strain - sys.float_info.epsilon,
                self.rect_strain,
                self.eta_c,
            ]
        )

    @property
    def _stress_values(self):
        """
        Return the stress values for the stress-strain curve.
        """

        return np.asarray([0.0, 0.0, self.f_c * self.alpha_2, self.f_c * self.alpha_2])

    def get_stress(self, strain: float) -> float:
        """
        Return the stress for a given strain.
        If the strain is outside the range of -ve ultimate strain / +ve ultimate strain,
        return 0.

        Parameters
        ----------
        strain : float
            The strain to calculate the stress for.

        Returns
        -------
        float
            The stress for the given strain.
        """

        return np.interp(
            strain, self._strain_values, self._stress_values, left=0.0, right=0.0
        )

    def plot_stress_strain(self):
        """
        Plot the stress-strain curve for the steel.
        """

        plt.plot(self._strain_values, self._stress_values, label="Stress vs Strain")
        plt.show()

    def __repr__(self):
        return f"{type(self).__name__}: f_c={self.f_c} MPa, eta_c={self.eta_c}"


class Steel:
    """
    Class to represent a steel material.

    The steel material is assumed to be linear elastic - perfectly plastic.
    """

    def __init__(
        self,
        *,
        f_sy: float,
        elastic_modulus: float = 200e3,
        eta_sy: float | None = None,
        eta_su: float = 0.05,
        ductility: Ductility | str = Ductility.N,
    ):
        """
        Initialise the steel material.

        Parameters
        ----------
        f_sy : float
            The yield strength of the steel.
        elastic_modulus : float
            The elastic modulus of the steel.
        eta_sy : float
            The yield strain of the steel.
        eta_su : float
            The ultimate strain of the steel.
            Default value is 0.05, which is the minimum ultimate strain required by
            AS4671 for N grade bar.
        ductility : Ductility
            The ductility of the steel.
            Default value is 'N', for N grade bar.
        """

        self._f_sy = f_sy
        self._elastic_modulus = elastic_modulus
        self._eta_sy = eta_sy
        self._eta_su = eta_su

        self._ductility = (
            Ductility(ductility) if isinstance(ductility, str) else ductility
        )

    @property
    def f_sy(self) -> float:
        """
        Return the yield strength of the steel.
        """

        return self._f_sy

    @property
    def elastic_modulus(self) -> float:
        """
        Return the elastic modulus of the steel.
        """

        return self._elastic_modulus

    @property
    def eta_sy(self) -> float:
        """
        Return the yield strain of the steel.
        """

        if self._eta_sy is None:
            return self.f_sy / self.elastic_modulus

        return self._eta_sy

    @property
    def eta_su(self) -> float:
        """
        Return the ultimate strain of the steel.
        """

        return self._eta_su

    @property
    def ductility(self) -> Ductility:
        """
        Return the ductility of the steel.
        """

        return self._ductility

    @property
    def _strain_values(self) -> np.ndarray:
        """
        Return the strain values for the stress-strain curve.
        """

        return np.asarray(
            [
                -self.eta_su,
                -self.eta_sy,
                0.0,
                self.eta_sy,
                self.eta_su,
            ]
        )

    @property
    def _stress_values(self) -> np.ndarray:
        """
        Return the stress values for the stress-strain curve.
        """

        return np.asarray([-self.f_sy, -self.f_sy, 0.0, self.f_sy, self.f_sy])

    def get_stress(self, strain: float) -> float:
        """
        Return the stress for a given strain.
        If the strain is outside the range of -ve ultimate strain / +ve ultimate strain,
        return 0.

        Parameters
        ----------
        strain : float
            The strain to calculate the stress for.

        Returns
        -------
        float
            The stress for the given strain.
        """

        return np.interp(
            strain, self._strain_values, self._stress_values, left=0.0, right=0.0
        )

    def plot_stress_strain(self):
        """
        Plot the stress-strain curve for the steel.
        """

        plt.plot(self._strain_values, self._stress_values, label="Stress vs Strain")
        plt.show()

    def __repr__(self):
        return (
            f"{type(self).__name__}: f_sy: {self.f_sy} MPa, "
            + f"ductility:'{self.ductility}', "
            + f"Yield Strain: {self.eta_sy:.3%}, "
            + f"Ultimate Strain: {self.eta_su:.0%}"
        )


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


def circle_area(diameter):
    """
    Calculate the area of a circle.

    :param dia: the diameter of the circle.
    """

    return pi * (diameter**2) / 4


def alpha_2(f_c, *, sect_type: SectType813 | str = SectType813.RECTANGULAR):
    """
    Calculate parameter alpha_2 as per AS3600-2018.

    :param f_c: The characteristic compressive strength of the concrete
    """

    if isinstance(sect_type, str):
        sect_type = SectType813(sect_type)

    if sect_type == SectType813.RECTANGULAR:
        multiplier = 1.00
    elif sect_type == SectType813.CIRCULAR:
        multiplier = 0.95
    else:
        multiplier = 0.90

    return max(0.67, 0.85 - 0.0015 * f_c) * multiplier


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
        The shear area of a single (set of) fitment in mm²
    f_sy : float
        The yield strength of the shear reinforcement.
    d_v : float
        The effective depth in shear in mm.
    theta_v : float
        The shear strut angle in radians
    s : float
        The spacing of shear fitments in mm.
        The default value is 1000 mm.
        If A_sv is given in mm² / m then s does not need to be provided.
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


def v_interface_s8_4_3(a_sf, f_sy, a_interface, mu, gp, k_co, f_ct):
    """
    Calculate the interface shear capacity as per AS3600 S8.4.3.

    Notes
    -----
    AS3600 relies on some values being provided in unusual units (kN/mm etc.).
    This function has been slightly modified to work with pressures where possible.

    Parameters
    ----------
    a_sf : float
        Area of reinforcement crossing the interface in mm².
        If reinforcement is not anchored fully,
        or crosses the interface at an angle the user should adjust
        the area of reinforcement accordingly (or f_sy).
    f_sy : float
        The yield strength of the reinforcement in MPa.
    a_interface : float
        The area of the interface, in mm².
    mu : float
        The friction coefficient of the interface.
    gp : float
        Permanent compressive stress on the interface if any.
        If tensile stress is present provide a -ve value.
    k_co : float
        The adhesion coefficient of the concrete.
    f_ct : float
        The characteristic tensile strength of the concrete in MPa.

    Returns
    -------
    float
    """

    return mu * ((a_sf * f_sy) / a_interface + gp) + k_co * f_ct


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


class RectBeam:
    """
    Class to represent a simple rectangular beam.

    Notes
    -----
    - At this point only top, bottom & side reinforcement will be considered.
    - Only rectangular sections will be considered.
    - All steel assumed to be the same grade.
    - All reo is assumed to be evenly distributed across the relevant face.

    For more complex sections recommend using the `concreteproperties` package.
    """

    def __init__(
        self,
        *,
        b: float,
        d: float,
        concrete: Concrete,
        steel: Steel,
        cover: float | dict[str, float],
        bot_dia: float | None = None,
        bot_no: float | None = None,
        top_dia: float | None = None,
        top_no: float | None = None,
        side_dia: float | None = None,
        side_no: float | None = None,
        shear_dia: float | None = None,
        shear_no_ligs: float | None = None,
        shear_spacing: float | None = None,
        shear_steel: Steel | None = None,
    ):
        """
        Initialize the RectBeam class.

        Parameters
        ----------
        b : float
            The width of the beam.
        d : float
            The depth of the beam.
        concrete : Concrete
            A Concrete material object.
        steel : Steel
            A Steel material object.
        cover : float | dict[str, float]
            The cover to the reinforcement.
            If a float is provided, it is assumed to be the same for all reinforcement.
            If a dictionary is provided, it should contain the keys
            "bot", "top" & "side".
        bot_dia : float
            The diameter of the bottom reinforcement.
        bot_no : float
            The number of bottom reinforcement bars.
        top_dia : float
            The diameter of the top reinforcement.
        top_no : float
            The number of top reinforcement bars.
        side_dia : float
            The diameter of the side reinforcement.
        side_no : float
            The number of side reinforcement bars.
        shear_dia : float
            The diameter of the shear reinforcement.
        shear_no_ligs : float
            The number of legs of shear reinforcement.
        shear_spacing : float
            The spacing of the shear reinforcement.
        shear_steel : Steel | None
            The shear reinforcement.
        """

        self._b = b
        self._d = d
        self._concrete = concrete
        self._steel = steel
        self._bot_dia = bot_dia
        self._bot_no = bot_no
        self._top_dia = top_dia
        self._top_no = top_no
        self._side_dia = side_dia
        self._side_no = side_no
        self._shear_dia = shear_dia
        self._shear_no_ligs = shear_no_ligs
        self._shear_spacing = shear_spacing
        self._shear_steel = shear_steel

        if isinstance(cover, (int, float)):
            self._bot_cover = float(cover)
            self._top_cover = float(cover)
            self._side_cover = float(cover)
        else:
            self._bot_cover = cover["bot"]
            self._top_cover = cover["top"]
            self._side_cover = cover["side"]

    @property
    def b(self):
        """
        Return the width of the beam.
        """

        return self._b

    @property
    def d(self):
        """
        Return the depth of the beam.
        """

        return self._d

    @property
    def concrete(self) -> Concrete:
        """
        Return a Concrete material object.
        """

        return self._concrete

    @property
    def steel(self) -> Steel:
        """
        Return a Steel material object.
        """

        return self._steel

    @property
    def modular_ratio(self):
        """
        Return the modular ratio of the steel & concrete.
        """

        return self.steel.elastic_modulus / self.concrete.elastic_modulus

    @property
    def top_dia(self):
        """
        Return the diameter of the top reinforcement.
        """

        if self._top_dia is None:
            return 0

        return self._top_dia

    @property
    def y_top(self):
        """
        Return the y-coordinate of the top reinforcement.
        """

        return self.d - self.top_cover - self.shear_dia - self.top_dia / 2

    @property
    def top_bar_area(self):
        """
        Return the area of the top reinforcement bars.
        """

        return circle_area(self.top_dia)

    @property
    def top_no(self):
        """
        Return the number of top reinforcement bars.
        """

        return self._top_no

    @property
    def area_steel_top(self):
        """
        Return the total area of the top reinforcement.
        """

        if self.top_no is None:
            return 0.0

        return self.top_bar_area * self.top_no

    @property
    def top_cover(self):
        """
        Return the cover to the top reinforcement.
        """

        return self._top_cover

    @property
    def top_bar_centres(self) -> list[tuple[float, float]]:
        """
        Return the centres of the top reinforcement bars.
        """

        if self.top_no is None or self.top_dia == 0:
            return []

        if self.top_no == 1:
            return [(0, self.y_top)]

        x_min = -self.b / 2 + (self.side_cover + self.shear_dia + self.top_dia / 2)
        x_max = self.b / 2 - (self.side_cover + self.shear_dia + self.top_dia / 2)
        centre_to_centre = (x_max - x_min) / (self.top_no - 1)

        return [(x_min + i * centre_to_centre, self.y_top) for i in range(self.top_no)]

    @property
    def bot_dia(self):
        """
        Return the diameter of the bottom reinforcement.
        """

        if self._bot_dia is None:
            return 0

        return self._bot_dia

    @property
    def y_bot(self):
        """
        Return the y-coordinate of the bottom reinforcement.
        """

        return self.bot_cover + self.shear_dia + self.bot_dia / 2

    @property
    def bot_bar_area(self):
        """
        Return the area of the bottom reinforcement bars.
        """

        return circle_area(self.bot_dia)

    @property
    def bot_no(self):
        """
        Return the number of bottom reinforcement bars.
        """

        return self._bot_no

    @property
    def area_steel_bot(self):
        """
        Return the total area of the bottom reinforcement.
        """

        if self.bot_no is None:
            return 0.0

        return self.bot_bar_area * self.bot_no

    @property
    def bot_cover(self):
        """
        Return the cover to the bottom reinforcement.
        """

        return self._bot_cover

    @property
    def bot_bar_centres(self) -> list[tuple[float, float]]:
        """
        Return the centres of the bottom reinforcement bars.
        """

        if self.bot_dia == 0 or self.bot_no is None:
            return []

        if self.bot_no == 1:
            return [(0, self.y_bot)]

        x_min = -self.b / 2 + (self.side_cover + self.shear_dia + self.bot_dia / 2)
        x_max = self.b / 2 - (self.side_cover + self.shear_dia + self.bot_dia / 2)
        centre_to_centre = (x_max - x_min) / (self.bot_no - 1)

        return [(x_min + i * centre_to_centre, self.y_bot) for i in range(self.bot_no)]

    @property
    def side_dia(self):
        """
        Return the diameter of the side reinforcement.
        """

        return self._side_dia

    @property
    def side_bar_area(self):
        """
        Return the area of the side reinforcement bars.
        """

        return circle_area(self.side_dia)

    @property
    def side_no(self):
        """
        Return the number of side reinforcement bars per side.
        """

        return self._side_no

    @property
    def area_steel_side(self):
        """
        Return the total area of the side reinforcement.
        """

        if self.side_no is None:
            return 0.0

        return self.side_bar_area * self.side_no * 2

    @property
    def side_cover(self):
        """
        Return the cover to the side reinforcement.
        """

        return self._side_cover

    @property
    def side_bar_centres(self) -> list[tuple[float, float]]:
        """
        Return the centres of the side reinforcement bars.
        """

        if self.side_dia == 0 or self.side_no is None:
            return []

        y_bot = self.y_bot + self.bot_dia / 2
        y_top = self.y_top - self.top_dia / 2

        gap = y_top - y_bot
        centre_to_centre = gap / (self.side_no + 1)

        x_left = -self.b / 2 + (self.side_cover + self.shear_dia + self.side_dia / 2)
        x_right = self.b / 2 - (self.side_cover + self.shear_dia + self.side_dia / 2)

        return [
            (x_left, y_bot + (i + 1) * centre_to_centre) for i in range(self.side_no)
        ] + [(x_right, y_bot + (i + 1) * centre_to_centre) for i in range(self.side_no)]

    @property
    def area_steel(self):
        """
        Return the total area of the reinforcement.
        """

        return self.area_steel_top + self.area_steel_bot + self.area_steel_side

    @property
    def shear_dia(self):
        """
        Return the diameter of the shear reinforcement.
        """

        if self._shear_dia is None:
            return 0

        return self._shear_dia

    @property
    def shear_bar_area(self):
        """
        Return the area of the shear reinforcement bars.
        """

        return circle_area(self.shear_dia)

    @property
    def shear_no_ligs(self):
        """
        Return the number of legs of shear reinforcement.
        """

        if self._shear_no_ligs is None:
            return 0

        return self._shear_no_ligs

    @property
    def area_steel_shear(self):
        """
        Return the total area of the shear reinforcement.
        """

        return self.shear_bar_area * self.shear_no_ligs

    @property
    def shear_spacing(self):
        """
        Return the spacing of the shear reinforcement.
        """

        return self._shear_spacing

    @property
    def shear_steel(self) -> Steel:
        """
        Return the shear reinforcement steel object.
        """

        if self._shear_steel is None:
            self._shear_steel = self.steel

        return self._shear_steel

    @property
    def area_gross(self):
        """
        Return the gross area of the beam.
        """

        return self.b * self.d

    @property
    def transformed_area(self):
        """
        Return the transformed area of the beam.
        """

        return (
            self.area_gross - self.area_steel
        ) + self.area_steel * self.modular_ratio
