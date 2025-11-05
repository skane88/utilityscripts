"""
Earthquake loads to AS1170.4
"""

from enum import StrEnum
from math import ceil, floor, log10
from numbers import Number
from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import cm

FILE_PATH = Path(__file__)
DEFAULT_DATA_PATH = FILE_PATH.parent / Path("as1170_4_data.xlsx")
STANDARD_DATA = {}


def init_standard_data(*, file_path: Path | None = None, overwrite: bool = False):
    global STANDARD_DATA

    if not overwrite and len(STANDARD_DATA) > 0:
        return

    if file_path is None:
        file_path = DEFAULT_DATA_PATH

        sheet_set = {
            "soil_descriptors",
            "soil_spectra",
            "k_p",
            "k_p_z_min",
        }

        STANDARD_DATA = {
            sheet: pl.read_excel(source=file_path, sheet_name=sheet)
            for sheet in sheet_set
        }


class SoilClass(StrEnum):
    """
    The type of soil
    """

    Ae = "Ae"
    Be = "Be"
    Ce = "Ce"
    De = "De"
    Ee = "Ee"


def k_p_data(
    *,
    p: float | np.ndarray,
) -> float | np.ndarray:
    """
    Determine the probability factor k_p for a given period as per AS1170.4 Table 3.1

    Notes
    -----
    - If p is greater than 2500 an error is returned.
    - If p is less than 100, the value for 1/100 is returned.

    Parameters
    ----------
    p : float
        The annual probabiltiy of exceedance.

    Returns
    -------
    float
        The probability factor k_p
    """

    init_standard_data()

    data = STANDARD_DATA["k_p"]
    data = data.sort("P")

    x = data["P"]
    y = data["k_p"]

    if isinstance(p, Number) and p > x.max():
        raise ValueError(
            "Probability of exceedance larger the bounds of Table 3.1. "
            + f" {p} > {x.max()}"
        )

    if isinstance(p, np.ndarray) and np.any(p > x.max()):
        raise ValueError(
            "Probability of exceedance larger than the bounds of Table 3.1. "
            + "Check input values."
        )

    k_p = np.interp(p, x, y)

    if isinstance(p, float | int):
        return float(k_p)

    return k_p


def k_p_z_min(*, p: float | np.ndarray) -> float | np.ndarray:
    """
    Determine the minimum value of k_p * z for a given probability of exceedance, in
    accordance with AS1170.4 Table 3.3.

    Notes
    -----
    - If p is greater than 2500 an error is returned.
    - If p is less than 500, the value for 1/500 is returned.

    Parameters
    ----------
    p : float
        The annual probability of exceedance.

    Returns
    -------
    float
        The minimum value of k_p * z
    """

    init_standard_data()

    data = STANDARD_DATA["k_p_z_min"]
    data = data.sort("P")

    x = data["P"]
    y = data["k_p_z_min"]

    if isinstance(p, Number) and p > x.max():
        raise ValueError(
            "Probability of exceedance larger the bounds of Table 3.1. "
            + f" {p} > {x.max()}"
        )

    if isinstance(p, np.ndarray) and np.any(p > x.max()):
        raise ValueError(
            "Probability of exceedance larger than the bounds of Table 3.1. "
            + "Check input values."
        )

    k_p_z_min = np.interp(p, x, y)

    if isinstance(p, float | int):
        return float(k_p_z_min)

    return k_p_z_min


def spectral_shape_factor(
    *, soil_class: SoilClass, period: float, min_period: bool = False
) -> float:
    """
    Calculate the spectral shape factor for a given soil type and period.

    Parameters
    ----------
    soil_class : SoilClass
        The type of soil
    period : float
        The period of the structure
    min_period : bool
        If True, enforce a minimum period of 0.1 seconds.

    Returns
    -------
    float
        The spectral shape factor
    """

    if period < 0:
        raise ValueError(f"Period must be greater than 0.0, got {period}.")

    if min_period:
        period = max(period, 0.1)

    init_standard_data()

    data = STANDARD_DATA["soil_spectra"]

    # first select only the row from the spectra data that we need.
    data = data.filter(pl.col("soil_class") == soil_class)
    data = data.filter(period <= pl.col("max_period"))
    data = data.filter(pl.col("max_period") == pl.col("max_period").min())

    # now pull the values into a dictionary
    values = data.to_dicts()[0]

    if period < values["min_period"]:
        raise ValueError(
            "Period is less than the expected minimum period for "
            + "which this equation is valid. "
            + f"{period} <= {values['min_period']}"
        )

    if period > values["max_period"]:
        raise ValueError(
            "Period is greater than the expected maximum period "
            + "for which this equation is valid. "
            + f"{period} >= {values['max_period']}"
        )

    # now calculate the spectral shape factor
    t = values["T"]
    c = values["C"]
    a = values["1/T"]
    b = values["1/T^2"]
    max_val = values["max_val"]

    ch = c if period == 0 else t * period + c + a * (1 / period) + b * (1 / period**2)

    return min(max_val, ch)


def k_p_z(*, p: float, z: float, min_kpz: bool = True) -> float:
    """
    Calculate the product of the probability factor and the site hazard design factor.

    Parameters
    ----------
    p : float
        The annual probability of exceedance.
    z : float
        The site hazard design factor.
    min_kpz : bool
        If True, enforce a minimum value of k_p * z.
        This is calculated with the function k_p_z_min.

    Returns
    -------
    float
        The product of the probability factor and the site hazard design factor.
    """

    k_p = k_p_data(p=p)
    k_p_z = k_p * z

    if min_kpz:
        k_p_z = max(k_p_z, k_p_z_min(p=p))

    return k_p_z


def plot_spectra(
    *,
    t_min: float = 0.0,
    t_max: float = 5.0,
    semi_log: bool = False,
    no_points: int = 201,
):
    """
    Plot the spectra for all soil types.

    Parameters
    ----------
    t_min : float
        The minimum period to plot. In s.
    t_max : float
        The maximum period to plot. In s.
    semi_log : bool
        Should the x-axis be semi-logarithmic?
        Note: if semi_log is True and t_min == 0, t_min is set to 0.01.
    no_points : int
        The no. of points to plot.
    """

    if semi_log and t_min == 0:
        warn("Use of t_min==0 will result in errors. t_min set to 0.01", stacklevel=2)
        t_min = 0.01

    periods = (
        np.logspace(floor(log10(t_min)), ceil(log10(t_max)), no_points)
        if semi_log
        else np.linspace(t_min, t_max, no_points)
    )
    cmap = cm.get_cmap("viridis", len(SoilClass))

    for i, soil_class in enumerate(SoilClass):
        y = [
            spectral_shape_factor(soil_class=soil_class, period=float(period))
            for period in periods
        ]

        plt.plot(periods, y, label=soil_class, color=cmap(i))

    plt.grid()
    plt.xlabel("Period T (s)")
    plt.ylabel("Spectral Ordinates Ch(T)")
    plt.ylim(0.0, 4.0)

    if semi_log:
        plt.xscale("log")
        plt.xlim(10 ** floor(log10(t_min)), 10 ** ceil(log10(t_max)))
    else:
        plt.xlim(t_min, t_max)

    plt.legend()
    plt.show()


def c_t(
    *,
    soil_class: SoilClass,
    period: float,
    k_p_z: float,
    min_period: bool = False,
) -> float:
    """
    The elastic site hazard spectrum, calculated from the spectral shape factor,
    the hazard design factor and the probability factor.

    Notes
    -----
    - This is equivalent to the lateral acceleration which would be
      experienced in an earthquake, as a percentage of gravity.
    - In real structures the force experienced is likely less than would
      be calculated from this value due to plastic deformation and other
      energy absorbing mechanisms. For the design force refer to function
      cd_t.

    Parameters
    ----------
    soil_class : SoilClass
        The type of soil
    period : float
        The period of the structure
    k_p_z : float
        The product of the probability factor and the site hazard design factor.
        This is required to be calculated by the caller external to this function,
        so that there is no confusion over what value of k_p x z is used in this
        function.
    min_period : bool
        If True, enforce a minimum period of 0.1 seconds.

    Returns
    -------
    float
        The elastic site hazard spectrum
    """

    ch_t = spectral_shape_factor(
        soil_class=soil_class, period=period, min_period=min_period
    )

    return k_p_z * ch_t


def cd_t(
    *,
    soil_class: SoilClass,
    period: float,
    k_p_z: float,
    s_p: float,
    mu: float,
    min_period: bool = False,
) -> float:
    """
    The horizontal design action coefficient.

    Notes
    -----
    - This is equivalent to the lateral acceleration experienced by the
      structure due to an earthquake, as a percentage of gravity.
    - This differs from the elastic site hazard spectrum in that it is
      scaled by the ductility factor and the structural peformance factor.
      This gives a more realistic value for the design force.

    Parameters
    ----------
    soil_class : SoilClass
        The type of soil
    period : float
        The period of the structure
    k_p_z : float
        The product of the probability factor and the site hazard design factor.
        This is required to be calculated by the caller external to this function,
        so that there is no confusion over what value of k_p x z is used in this
        function.
    s_p : float
        The structural performance factor
    mu : float
        The ductility factor
    min_period : bool
        If True, enforce a minimum period of 0.1 seconds.

    Returns
    -------
    float
        The horizontal design action coefficient
    """

    return c_t(
        soil_class=soil_class,
        period=period,
        k_p_z=k_p_z,
        min_period=min_period,
    ) * (s_p / mu)
