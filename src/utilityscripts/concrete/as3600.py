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


class S813SectType(StrEnum):
    """
    An Enum to represent the 3x types of section referred to in S8.1.3 note 2.
    """

    RECTANGULAR = "rectangular"
    CIRCULAR = "circular"
    OTHER = "other"


class S816SectType(StrEnum):
    """
    An Enum to represent the diferent types of sections in S8.1.6.1
    """

    RECTANGULAR = "rectangular"
    TEE_WEB_DOWN = "tee_web_down"
    TEE_WEB_UP = "tee_web_up"


class Concrete:
    """
    Class to represent a concrete material.
    """

    def __init__(
        self,
        f_c: float,
        *,
        density: float = 2400,
        epsilon_c: float = 0.003,
        poisson_ratio: float = 0.2,
        f_cm: float | None = None,
        f_cmi: float | None = None,
        elastic_modulus: float | None = None,
        sect_type: S813SectType = S813SectType.RECTANGULAR,
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
        epsilon_c : float
            The maximum compressive strain of the concrete.
            Default value is 0.003.
        poisson_ratio : float
            The Poisson's ratio of the concrete.
            Default value is 0.2.
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
        self._epsilon_c = epsilon_c
        self._poisson_ratio = poisson_ratio
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
    def epsilon_c(self):
        """
        Return the maximum compressive strain of the concrete.
        """

        return self._epsilon_c

    @property
    def poisson_ratio(self):
        """
        Return the Poisson's ratio of the concrete.
        """

        return self._poisson_ratio

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
            self._f_cm = s3_1_2_get_f_cm(self._f_c)

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
            self._f_cmi = s3_1_2_get_f_cmi(self._f_c)

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

        return s3_1_1_3_f_ctf(self.f_c)

    @property
    def f_ct(self):
        """
        Calculate the characteristic tensile strength of the concrete
        as per AS3600 S3.1.1.3.
        """

        return s3_1_1_3_f_ct(self.f_c)

    @property
    def alpha_2(self):
        """
        Calculate parameter alpha_2 as per AS3600-2018.
        """

        return s8_1_3_alpha_2(f_c=self.f_c, sect_type=self._sect_type)

    @property
    def gamma(self):
        """
        Calculate parameter gamma as per AS3600-2018.
        """

        return s8_1_3_gamma(self.f_c)

    @property
    def rect_strain(self):
        """
        Return the minimum strain for the rectangular stress block
        """

        return self.epsilon_c * (1 - self.gamma)

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
                self.epsilon_c,
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


def s3_1_2_get_f_cm(f_c):
    """
    Determine the mean compressive strength, based on the values given in
    AS3600-2018 Table 3.1.2
    """

    f_c_vals = np.asarray(STANDARD_GRADES["f_c"])
    f_cm_vals = np.asarray(STANDARD_GRADES["f_cm"])

    return np.interp(f_c, f_c_vals, f_cm_vals)


def s3_1_2_get_f_cmi(f_c):
    """
    Determine the mean in-situ compressive strength, based on the values given in
    AS3600-2018 Table 3.1.2
    """

    f_c_vals = np.asarray(STANDARD_GRADES["f_c"])
    f_cmi_vals = np.asarray(STANDARD_GRADES["f_cmi"])

    return np.interp(f_c, f_c_vals, f_cmi_vals)


def s3_1_1_3_f_ctf(f_c):
    """
    Determine the characteristic flexural strength of the concrete, based on S3.1.1.3 of
    AS3600-2018.

    Parameters
    ----------
    f_c : float
        The characteristic compressive strength of the concrete.

    Returns
    -------
    float
        The characteristic flexural strength of the concrete.
    """

    return 0.6 * (f_c**0.5)


def s3_1_1_3_f_ct(f_c):
    """
    Determine the characteristic tensile strength of the concrete, based on S3.1.1.3 of
    AS3600-2018.

    Parameters
    ----------
    f_c : float
        The characteristic compressive strength of the concrete.

    Returns
    -------
    float
        The characteristic tensile strength of the concrete.
    """

    return 0.36 * (f_c**0.5)


def s8_1_3_alpha_2(
    f_c: float, *, sect_type: S813SectType | str = S813SectType.RECTANGULAR
) -> float:
    """
    Calculate parameter alpha_2 as per AS3600-2018.

    Parameters
    ----------
    f_c : float
        The characteristic compressive strength of the concrete
    sect_type : SectType813
        Is the section Rectangular or Circular.
    """

    if isinstance(sect_type, str):
        sect_type = S813SectType(sect_type)

    if sect_type == S813SectType.RECTANGULAR:
        multiplier = 1.00
    elif sect_type == S813SectType.CIRCULAR:
        multiplier = 0.95
    else:
        multiplier = 0.90

    return max(0.67, 0.85 - 0.0015 * f_c) * multiplier


def s8_1_3_gamma(f_c: float) -> float:
    """
    Calculate parameter gamma as per AS3600-2018 Section 8.1.3.

    Parameters
    ----------
    f_c : float
        The characteristic compressive strength of the concrete.
    """

    return max(0.67, 0.97 - 0.0025 * f_c)


def generate_rectilinear_block(*, f_c: float, max_compression_strain: float = 0.003):
    """
    Generate a rectilinear stress-strain curve as required by AS3600.

    Parameters
    ----------
    Parameters
    ----------
    f_c : float
        The characteristic compressive strength of the concrete. In MPa.
    max_compression_strain : float
        The maximum allowable compressive strain in the concrete. According to AS3600
        S8.1.3 this is 0.003.
    """

    gamma_val = s8_1_3_gamma(f_c)
    min_strain = max_compression_strain * (1 - gamma_val)

    compressive_strength = s8_1_3_alpha_2(f_c) * f_c

    return [
        [0, min_strain, max_compression_strain],
        [0, compressive_strength, compressive_strength],
    ]


def s8_1_6_1_m_uo_min(*, z, f_ct_f, p_e=0, a_g=0, e=0):
    """
    Calculate the minimum required moment capacity as per AS3600-2018 S8.1.6.1.

    Parameters
    ----------
    z : float
        The uncracked section modulus, taken at the face of the section at which
        cracking occurs.
    f_ct_f : float
        The characteristic flexural strength of the concrete.
    p_e : float
        The effective prestress force, accounting for losses.
    a_g : float
        The gross area.
    e : float
        The prestress eccentricity from the centroid of the uncracked section.
    """

    return 1.2 * (z * (f_ct_f + p_e / a_g) + p_e * e)


def s8_1_6_1_a_st_min(
    *,
    d_beam,
    d_reo,
    b_w,
    f_ct_f,
    f_sy,
    sect_type: S816SectType = S816SectType.RECTANGULAR,
):
    """
    Calculate the deemed-to-comply minimum area of tensile reinforcement required
    as per AS3600-2018 S8.1.6.1.

    Parameters
    ----------
    d_beam : float
        The depth of the beam. In mm.
    d_reo : float
        The centroid of tensile reinforcement. In mm.
    b_w : float
        The width of the beam or beam web. In mm.
    f_ct_f : float
        The characteristic flexural strength of the concrete. In MPa.
    f_sy : float
        The yield strength of the reinforcement. In MPa.
    sect_type : S816SectType
        The type of section.
    """

    match sect_type:
        case S816SectType.RECTANGULAR:
            return 0.20 * ((d_beam / d_reo) ** 2) * (f_ct_f / f_sy) * b_w * d_reo
        case S816SectType.TEE_WEB_DOWN:
            raise NotImplementedError("Tee web down sections are not yet implemented.")
        case S816SectType.TEE_WEB_UP:
            raise NotImplementedError("Tee web up sections are not yet implemented.")
        case _:
            raise ValueError(f"Invalid section type: {sect_type}")


def m_uo():
    """
    Method to calculate the concrete capacity of a rectangular beam as per AS3600.
    """

    pass


def s8_2_1_7_a_svmin(*, f_c, f_sy, b_v, s: float = 1000):
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


def s8_2_3_1_v_u(
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

    return s8_2_4_1_v_uc(f_c=f_c, k_v=k_v, b_v=b_v, d_v=d_v) + s8_2_5_2_v_us(
        a_sv=a_sv, f_sy=f_sy, d_v=d_v, theta_v=theta_v, s=s, a_v=a_v
    )


def s8_2_3_2_v_u_max(
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


def s8_2_4_1_v_uc(*, k_v, b_v, d_v, f_c):
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


def s8_2_4_2_theta_v(*, eta_x, return_radians: float = True):
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


def s8_2_4_2_k_v(*, f_c, a_sv, a_svmin, eta_x, d_v, d_g: float = 20.0):
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


def s8_2_4_2_eta_x(
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


def s8_2_4_3_theta_v(*, return_radians: bool = True):
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


def s8_2_4_3_k_v(*, a_sv, a_svmin, d_v):
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


def s8_2_5_2_v_us(*, a_sv, f_sy, d_v, theta_v, s: float = 1000, a_v: float = pi / 2):
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


def s8_4_3_v_interface(a_sf, f_sy, a_interface, mu, gp, k_co, f_ct):
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


def s13_1_2_c_d(
    *, bar_spacing, cover, bar_type: str = "straight", narrow_element: bool = True
):
    """
    Calculate the cover and spacing parameter c_d as per AS3600 S13.1.2

    Parameters
    ----------
    bar_spacing: The spacing between bars.
    cover:
        The cover to the bars.
        See figure 13.1.2.2 to determine which cover is appropriate.
    bar_type:
        The type of bar, either "straight", "hooked" or "looped"
    narrow_element:
        Is the element narrow (e.g. beam web, column)
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


def s13_1_2_2_k_3(
    *, d_b, bar_spacing, cover, bar_type: str = "straight", narrow_element: bool = True
):
    """
    Calculate parameter k_3 as per AS3600 S13.1.2.2

    Parameters
    ----------
    d_b:
        bar diamter.
        Units should be consistent with bar_spacing and cover.
    bar_spacing:
        The spacing between bars.
    cover:
        The cover to the bars.
        See figure 13.1.2.2 to determine which cover is appropriate.
    bar_type:
        The type of bar, either "straight", "hooked" or "looped"
    narrow_element:
        Is the element narrow (e.g. beam web, column)
        or wide (e.g. slab, inner bars of a band beam etc.)
    """

    c_d_calc = s13_1_2_c_d(
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


def s13_1_2_3_l_syt(*, f_c, f_sy, d_b, k_1=1.0, k_3=1.0, k_4=1.0, k_5=1.0):
    """
    Calculate the development length of a bar as per AS3600 S13.1.2.3

    Parameters
    ----------
    f_c:
        The concrete characteristic compressive strength, f'c. In MPa.
    f_sy:
        The steel yield strength (in MPa)
    d_b:
        The diameter of the bar (in mm)
    k_1:
        The member depth parameter.
    k_3:
        The bar spacing parameter.
    k_4:
        Transverse reinforcement parameter.
    k_5:
        Transverse stress parameter.

    Returns
    -------
    The development length in mm.
    """

    k_2 = (132 - d_b) / 100
    k_prod = max(0.7, k_3 * k_4 * k_5)
    l_calc = 0.5 * k_1 * k_prod * f_sy * d_b / (k_2 * f_c**0.5)
    l_min = 0.058 * f_sy * k_1 * d_b

    return max(l_calc, l_min)


def s13_2_2_l_syt_lap(
    *, l_syt: float, f_sy: float, d_b: float, k_1: float = 1.3, k_7: float = 1.25
):
    """
    Calculate the lap length of a bar in tension as per AS3600 S13.2.2.

    Parameters
    ----------
    l_syt: float
        The basic development length of the bar in tension. In mm.
    f_sy: float
        The yield strength of the bar in MPa.
    d_b: float
        The bar diameter in mm.
    k_1: float
        The parameter for the depth of concrete below the bar. Conservatively 1.3.
    k_7: float
        The parameter for the provided area of steel. Conservatively 1.25.

    Returns
    -------
    float
        The required lap length in mm.
    """

    lap_1 = k_7 * l_syt
    lap_2 = 0.058 * f_sy * k_1 * d_b

    return max(lap_1, lap_2)
