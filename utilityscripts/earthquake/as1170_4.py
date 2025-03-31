"""
Earthquake loads to AS1170.4
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

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


@lru_cache(maxsize=None)
def spectral_shape_factor(
    *, soil_type: SoilClass, period: float, min_period: bool = False
) -> float:
    """
    Calculate the spectral shape factor for a given soil type and period.

    Parameters
    ----------
    soil_type : SoilType
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
    data = data.filter(pl.col("soil_type") == soil_type)
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


def plot_spectra():
    """
    Plot the spectra for all soil types.
    """

    periods = np.linspace(0.0, 5.0, 200)

    for soil_type in SoilClass:
        y = [
            spectral_shape_factor(soil_type=soil_type, period=float(period))
            for period in periods
        ]

        plt.plot(periods, y, label=soil_type)

    plt.grid()
    plt.xlabel("Period T (s)")
    plt.ylabel("Spectral Ordinates Ch(T)")
    plt.xlim(0.0, 5.0)
    plt.ylim(0.0, 4.0)
    plt.legend()
    plt.show()


def c_t(
    *,
    soil_type: SoilClass,
    period: float,
    k_p: float,
    z: float,
    min_kpz: float = 0.08,
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
    soil_type : SoilType
        The type of soil
    period : float
        The period of the structure
    k_p : float
        The probability factor
    z : float
        The site hazard design factor
    min_kpz : float
        The minimum value of k_p * z.
        AS1170.4 requires a minimum value of 0.08.
    min_period : bool
        If True, enforce a minimum period of 0.1 seconds.

    Returns
    -------
    float
        The elastic site hazard spectrum
    """

    kp_z = max(min_kpz, k_p * z)
    ch_t = spectral_shape_factor(
        soil_type=soil_type, period=period, min_period=min_period
    )

    return kp_z * ch_t


def cd_t(
    *,
    soil_type: SoilClass,
    period: float,
    k_p: float,
    z: float,
    s_p: float,
    mu: float,
    min_kpz: float = 0.08,
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
    soil_type : SoilType
        The type of soil
    period : float
        The period of the structure
    k_p : float
        The probability factor
    z : float
        The site hazard design factor
    s_p : float
        The structural performance factor
    mu : float
        The ductility factor
    min_kpz : float
        The minimum value of k_p * z.
        AS1170.4 requires a minimum value of 0.08.
    min_period : bool
        If True, enforce a minimum period of 0.1 seconds.

    Returns
    -------
    float
        The horizontal design action coefficient
    """

    return c_t(
        soil_type=soil_type,
        period=period,
        k_p=k_p,
        z=z,
        min_kpz=min_kpz,
        min_period=min_period,
    ) * (s_p / mu)
