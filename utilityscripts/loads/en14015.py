"""
Contains methods and classes for calculating seismic loads on tanks as per EN14015.
"""

from enum import StrEnum
from functools import lru_cache
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.interpolate import interp1d

from utilityscripts.result import Result


class SoilType(StrEnum):
    A = "A"
    B = "B"
    C = "C"


class EN14015:
    def __init__(
        self,
        *,
        h_l: float,
        h_t: float,
        r_e: float,
        t_shell: float,
        m_roof: float,
        rho_liquid: float,
        a_seismic: float,
        soil_profile: SoilType,
        rho_shell: float = 7850.0,
    ):
        """

        Initialise a tank to design to EN14015-2004.

        Notes
        -----
        - Initially only seismic loads are intended to be supported.
        - The tank shell is assumed to be a constant thickness.

        Parameters
        ----------
        h_l : float
            The total height of the tank shell. In m.
        h_t : float
            The filling height of the tank.
            The height to the top of the curb angle or overflow that limits the
            filling height.
            In m.
        r_e : float
            The external radius of the tank shell. In m.
        t_shell : float
            The thickness of the tank shell. In m.
        m_roof : float
            The mass of the roof. In kg.
        rho_liquid : float
            The density of the liquid. In kg/m^3.
        a_seismic : float
            The seismic acceleration coefficient for inertial movement of the tank.
            In m/s^2.
        soil_profile: str
            The soil profile type. Should be one of:
            A, B, C
        rho_shell : float
            The density of the tank shell.
            Default is 7850 kg/m^3 for steel.
        """

        self._h_l = h_l
        self._h_t = h_t
        self._r_e = r_e
        self._t_shell = t_shell
        self._m_roof = m_roof
        self._rho_liquid = rho_liquid
        self._a_seismic = a_seismic
        self._soil_profile = soil_profile
        self._rho_shell = rho_shell

    @property
    def h_l(self) -> float:
        """
        The height of the tank shell.
        """

        return self._h_l

    @property
    def h_t(self) -> float:
        """
        The filling height of the tank.
        """

        return self._h_t

    @property
    def r_e(self) -> float:
        """
        The external radius of the tank shell.
        """

        return self._r_e

    @property
    def r_i(self) -> float:
        """
        The internal radius of the tank shell.
        """

        return self._r_e - self._t_shell

    @property
    def t_shell(self) -> float:
        """
        The thickness of the tank shell.
        """

        return self._t_shell

    @property
    def m_roof(self) -> float:
        """
        The mass of the roof.
        """

        return self._m_roof

    @property
    def rho_liquid(self) -> float:
        """
        The density of the liquid.
        """

        return self._rho_liquid

    @property
    def a_seismic(self) -> float:
        """
        The seismic acceleration coefficient for inertial movement of the tank.
        """

        return self._a_seismic

    @property
    def soil_profile(self) -> SoilType:
        """
        The soil profile type.
        """

        return self._soil_profile

    @property
    def rho_shell(self) -> float:
        """
        The density of the tank shell.
        """

        return self._rho_shell

    @property
    def d_e(self) -> float:
        """
        The external diameter of the tank shell.
        """

        return 2 * self.r_e

    @property
    def d_i(self) -> float:
        """
        The internal diameter of the tank shell.
        """

        return 2 * self.r_i

    @property
    def _a_liquid(self) -> float:
        """
        The area of the liquid in the tank.
        """

        return pi * self.r_i**2

    @property
    def v_liquid(self) -> float:
        """
        The volume of the liquid in the tank.
        """

        return self._a_liquid * self.h_t

    @property
    def m_liquid(self) -> float:
        """
        The mass of the liquid in the tank.
        """

        return self.rho_liquid * self.v_liquid

    @property
    def _a_shell(self) -> float:
        """
        The area of the tank shell.
        """

        return pi * self.r_e**2 - pi * self.r_i**2

    @property
    def m_shell(self) -> float:
        """
        The mass of the tank shell.
        """

        return self.rho_shell * self._a_shell * self.h_l

    @property
    def m_tank(self) -> float:
        """
        The total mass of the tank.
        """

        return self.m_shell + self.m_roof

    @property
    def m_total(self) -> float:
        """
        The total mass of tank + contents
        """

        return self.m_tank + self.m_liquid

    @property
    def d_h_t_ratio(self) -> Result:
        """
        The ratio of the contents diameter to the filling height of the tank.
        """

        return Result(
            self.d_i / self.h_t,
            description="Ratio of liquid diameter to filling height",
            eqn="d_i / h_t",
            inputs={"d_i": self.d_i, "h_t": self.h_t},
        )

    @property
    def t_1_ratio(self) -> float:
        """
        The fraction of the tank contents that behaves in an inertial manner.
        """

        return float(get_ratios(d_h_t_ratio=self.d_h_t_ratio.value)["t1-tt"])

    @property
    def t_1(self) -> Result:
        """
        The fraction of the tank contents that behaves in an inertial manner.
        """

        return Result(
            self.t_1_ratio * self.m_liquid,
            description="Fraction of liquid behaving in an inertial manner",
            variable="T_1",
            eqn="T_1/m_liquid Ratio x m_liquid",
            inputs={"m_liquid": self.m_liquid, "T_1 / m_liquid Ratio": self.t_1_ratio},
        )

    @property
    def t_2_ratio(self) -> float:
        """
        The fraction of the tank contents that behaves in a sloshing or
        convective manner.
        """

        return float(get_ratios(d_h_t_ratio=self.d_h_t_ratio.value)["t2-tt"])

    @property
    def t_2(self) -> float:
        """
        The mass of the tank contents that behaves in a sloshing or
        convective manner.
        """

        return self.t_2_ratio * self.m_liquid

    @property
    def x_1_ratio(self) -> float:
        """
        The height from the bottom of the tank at which the inertia load of the
        liquid acts, as a fraction of h_t.
        """

        return float(get_ratios(d_h_t_ratio=self.d_h_t_ratio.value)["x1-ht"])

    @property
    def x_2_ratio(self) -> float:
        """
        The height from the bottom of the tank at which the sloshing or
        convective load of the liquid acts, as a fraction of h_t.
        """

        return float(get_ratios(d_h_t_ratio=self.d_h_t_ratio.value)["x2-ht"])

    @property
    def x_1(self) -> float:
        """
        The height from the bottom of the tank at which the inertia load of the
        liquid acts. In m.
        """

        return self.x_1_ratio * self.h_t

    @property
    def x_2(self) -> float:
        """
        The height from the bottom of the tank at which the sloshing or
        convective load of the liquid acts. In m.
        """

        return self.x_2_ratio * self.h_t

    @property
    def x_t(self) -> float:
        """
        The height from the bottom of the tank at which the shell load acts.
        In m.
        """

        return self.h_l / 2

    @property
    def k_s(self) -> float:
        """
        The sloshing acceleration factor.
        """

        return float(get_ratios(d_h_t_ratio=float(self.d_h_t_ratio))["ks"])

    @property
    def j(self) -> float:
        """
        The soil profile factor.
        """

        return {SoilType.A: 1.0, SoilType.B: 1.2, SoilType.C: 1.5}[self.soil_profile]

    @property
    def t_s(self) -> float:
        """
        The sloshing period for the first natural mode.
        """

        return 1.8 * self.k_s * self.d_i**0.5

    @property
    def a_sloshing(self) -> float:
        """
        The sloshing acceleration.
        """

        if self.t_s <= 4.5:  # noqa: PLR2004
            return 1.25 * self.a_seismic * self.j / self.t_s

        return 5.625 * self.a_seismic * self.j / self.t_s**2

    @property
    def eq_t(self) -> float:
        """
        The seismic load for the weight of the tank.
        In kN.
        """

        return self.a_seismic * self.m_shell / 1000

    @property
    def eq_r(self) -> float:
        """
        The seismic load for the weight of the roof.
        In kN.
        """

        return self.a_seismic * self.m_roof / 1000

    @property
    def eq_1(self) -> float:
        """
        The seismic load for the weight of the contents that move in an inertial manner.
        In kN.
        """

        return self.a_seismic * self.t_1 / 1000

    @property
    def eq_2(self) -> float:
        """
        The seismic load for the weight of the contents that move in a sloshing or
        convective manner.
        In kN.
        """

        return self.a_sloshing * self.t_2 / 1000

    @property
    def eq_total(self) -> float:
        """
        The total seismic load.
        In kN.
        """

        return self.eq_t + self.eq_r + self.eq_1 + self.eq_2

    @property
    def meq_t(self) -> float:
        """
        The seismic moment for the tank wall
        In kNm.
        """

        return self.eq_t * self.x_t

    @property
    def meq_r(self) -> float:
        """
        The seismic moment for the roof
        In kNm.
        """

        return self.eq_r * self.h_l

    @property
    def meq_1(self) -> float:
        """
        The seismic moment for the contents that move in an inertial manner.
        In kNm.
        """

        return self.eq_1 * self.x_1

    @property
    def meq_2(self) -> float:
        """
        The seismic moment for the contents that move in a sloshing or
        convective manner.
        In kNm.
        """

        return self.eq_2 * self.x_2

    @property
    def meq_total(self) -> float:
        """
        The total seismic moment.
        In kNm.
        """

        return self.meq_t + self.meq_r + self.meq_1 + self.meq_2


@lru_cache(maxsize=None)
def _get_ratio_data() -> dict[str, pl.DataFrame]:
    """
    Get the ratio data from the Excel file.

    Notes
    -----
    This function is cached to avoid multiple calls into Excel.

    Returns
    -------
    dict[str, pl.DataFrame]
        A dictionary of the ratios, with keys:
        - t1-tt : The fraction of the tank contents that behaves in an
            inertial manner.
        - t2-tt : The fraction of the tank contents that behaves in a
            sloshing or convective manner.
        - x1-ht : The height from the bottom of the tank at which the inertia
            load of the liquid acts, as a fraction of h_t.
        - x2-ht : The height from the bottom of the tank at which the sloshing or
            convective load of the liquid acts, as a fraction of h_t.
        - ks : The sloshing parameter.
    """

    sheet_sets = {"t1-tt", "t2-tt", "x1-ht", "x2-ht", "ks"}
    file_path = Path(__file__).parent / Path("en14015_data.xlsx")

    return {
        sheet: pl.read_excel(source=file_path, sheet_name=sheet) for sheet in sheet_sets
    }


def plot_t1_t2_data():
    """
    Plot the t1-tt data.

    Notes
    -----
    Primarily provided for debugging purposes.
    """

    curves = {"t1-tt": "T_1 / m_liquid", "t2-tt": "T_2 / m_liquid"}

    for curve in curves:
        data = _get_ratio_data()[curve]

        plt.scatter(data["x"], data["y"], label=f"{curves[curve]} (data)")

    x = np.linspace(0, 8, 100)

    for curve in curves:
        interpolator = _get_ratio_interpolators()[curve]

        plt.plot(x, interpolator(x), label=f"{curves[curve]} (interpolated)")

    plt.xlim(0, 8)
    plt.ylim(0, 1)
    plt.grid()

    plt.legend()
    plt.show()


@lru_cache(maxsize=None)
def _get_ratio_interpolators() -> dict[str, interp1d]:
    """
    Get the ratio data from the Excel file and set up a
    series of SciPy interpolators.

    Notes
    -----
    This function is cached to avoid multiple calls into Excel.

    Returns
    -------
    dict[str, interp1d]
        A dictionary of the ratios, with keys:
        - t1-tt : The fraction of the tank contents that behaves in an
            inertial manner.
        - t2-tt : The fraction of the tank contents that behaves in a
            sloshing or convective manner.
        - x1-ht : The height from the bottom of the tank at which the inertia
            load of the liquid acts, as a fraction of h_t.
        - x2-ht : The height from the bottom of the tank at which the sloshing or
            convective load of the liquid acts, as a fraction of h_t.
        - ks : The sloshing parameter.
    """

    sheet_sets = {"t1-tt", "t2-tt", "x1-ht", "x2-ht", "ks"}
    file_path = Path(__file__).parent / Path("en14015_data.xlsx")

    ratios = {}

    for sheet in sheet_sets:
        data = pl.read_excel(source=file_path, sheet_name=sheet)

        x_vals = np.asarray(data["x"])
        y_vals = np.asarray(data["y"])

        interp = interp1d(x_vals, y_vals, kind="cubic", fill_value="extrapolate")

        ratios[sheet] = interp

    return ratios


@lru_cache(maxsize=None)
def get_ratios(*, d_h_t_ratio: float):
    """
    Get the fractions of tank contents that behave in the inertial
    and convection modes and their heights of application from
    Figure G.1 & G.2 of EN14015-2004.

    Gets the sloshing parameter k_s from figure G.3 of EN14015-2004.

    Notes
    -----
    This function is cached to avoid interpolating multiple times.

    Parameters
    ----------
    d_h_t_ratio : float
        The ratio of the diameter to the filling height of the tank.

    Returns
    -------
    dict[str, float]
        A dictionary of the ratios, with keys:
        - t1-tt : The fraction of the tank contents that behaves in an
            inertial manner.
        - t2-tt : The fraction of the tank contents that behaves in a
            sloshing or convective manner.
        - x1-ht : The height from the bottom of the tank at which the inertia
            load of the liquid acts, as a fraction of h_t.
        - x2-ht : The height from the bottom of the tank at which the sloshing or
            convective load of the liquid acts, as a fraction of h_t.
        - ks : The sloshing acceleration factor.
    """

    ratio_interpolators = _get_ratio_interpolators()

    ratios = {}

    for ratio_name, interpolator in ratio_interpolators.items():
        ratios[ratio_name] = interpolator(d_h_t_ratio)

    return ratios
