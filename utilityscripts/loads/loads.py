"""
Utilities for working with loads
"""

from enum import StrEnum
from functools import lru_cache
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from utilityscripts.plotting import AGILITUS_COLORS


class RoofType(StrEnum):
    ACCESSIBLE = "accessible"
    ACCESSIBLEFROMGROUND = "accessible from ground"
    OTHER = "other"


def pipe_load(*, pipe_density, content_density, outer_diameter, inner_diameter):
    """
    Calculate loads in a pipe.

    :param pipe_density: The density of the pipe.
    :param content_density: The density of the contents.
    :param outer_diameter: The outer diameter.
    :param inner_diameter: The inner diameter.
    :return: A dictionary of masses.
    """

    total_area = 0.25 * pi * outer_diameter**2
    inner_area = 0.25 * pi * inner_diameter**2

    pipe_area = total_area - inner_area

    empty_weight = pipe_area * pipe_density
    content_weight = inner_area * content_density
    full_weight = empty_weight + content_weight

    return {
        "empty_weight": empty_weight,
        "content_weight": content_weight,
        "full_weight": full_weight,
    }


def roof_load_as1170_1(
    roof_type: RoofType | str, area: float | None
) -> tuple[float, float]:
    """
    Calculate roof load as per AS1170.1 S3.5

    Notes
    -----
    Ignores the distinction between structure and cladding
    that AS1170 allows for OTHER type roofs.

    Parameters
    ----------
    roof_type : RoofType | str
        What sort of roof is it?
    area : float | None
        The tributary area.

    Returns
    -------
    tuple[float, float]

    The loads to apply as (pressure in kPa, point load in kN)
    """

    if roof_type == RoofType.ACCESSIBLE:
        return 1.5, 1.8

    if roof_type == RoofType.ACCESSIBLEFROMGROUND:
        return 1.0, 1.8

    if roof_type != RoofType.OTHER:
        raise ValueError(
            "Roof type should be a valid RoofType Enum or equivalent string."
        )

    return max(0.25, 1.8 / area + 0.12), 1.4


_DATA_PATH = Path(Path(__file__).parent)
CRANE_DATA_FILE = "crane_data.xlsx"


class HoistClass(StrEnum):
    HC1 = "HC1"
    HC2 = "HC2"
    HC3 = "HC3"
    HC4 = "HC4"


class DriveClass(StrEnum):
    HD1 = "HD1"
    HD2 = "HD2"
    HD3 = "HD3"
    HD4 = "HD4"
    HD5 = "HD5"


@lru_cache(maxsize=None)
def _get_beta_2_data() -> pl.DataFrame:
    return pl.read_excel(_DATA_PATH / Path(CRANE_DATA_FILE), sheet_name="beta_2")


@lru_cache(maxsize=None)
def _get_phi_2_data() -> pl.DataFrame:
    return pl.read_excel(_DATA_PATH / Path(CRANE_DATA_FILE), sheet_name="phi_2_min")


def get_hoist_class(*, delta: float) -> HoistClass:
    """
    Get the hoist class. The hoist class is assigned based on the elastic stiffness
    of the crane. This is approximated by the characteristic deflection, delta.

    Parameters
    ----------
    delta : float
        The characteristic deflection.

    Returns
    -------
    HoistClass
        The hoist class.
    """

    beta_2_data = _get_beta_2_data()
    beta_2_data = beta_2_data.filter(pl.col("delta_max") > delta)
    beta_2_data = beta_2_data.filter(pl.col("delta_min") <= delta)

    return HoistClass(beta_2_data["hoist_class"][0])


def get_beta_2(*, hoist_class: HoistClass):
    """
    Get the value of beta_2 for a given hoisting class.

    Parameters
    ----------
    hoist_class : HoistClass
        The hoist class.

    Returns
    -------
    float
        The value of beta_2.
    """

    beta_2_data = _get_beta_2_data()
    beta_2_data = beta_2_data.filter(pl.col("hoist_class") == hoist_class)
    return beta_2_data["beta_2"][0]


def get_phi_2_min(*, hoist_class: HoistClass, drive_class: DriveClass):
    """
    Get the minimum value of phi_2 for a given hoist class and drive class.

    Parameters
    ----------
    hoist_class : HoistClass
        The hoist class.
    drive_class : DriveClass
        The drive class.

    Returns
    -------
    float
        The minimum value of phi_2.
    """

    phi_2_data = _get_phi_2_data()
    phi_2_data = phi_2_data.filter(pl.col("hoist_class") == hoist_class)
    phi_2_data = phi_2_data.filter(pl.col("drive_class") == drive_class)
    return phi_2_data["phi_2_min"][0]


def phi_2(*, v_h: float, hoist_class: HoistClass, drive_class: DriveClass):
    """
    Calculate the dynamic hoisting factor, phi_2, as per S6.1.2.1 of AS5221.1-2021.

    Parameters
    ----------
    v_h : float
        The hoist speed.
    hoist_class : HoistClass
        The hoist class.
    drive_class : DriveClass
        The drive class.

    Returns
    -------
    float
        The dynamic hoisting factor.
    """

    beta_2 = get_beta_2(hoist_class=hoist_class)
    phi_2_min = get_phi_2_min(hoist_class=hoist_class, drive_class=drive_class)

    return phi_2_min + beta_2 * v_h


def plot_phi_2(drive_class: DriveClass):
    v_h = np.linspace(0, 2, 100)

    ax = plt.gca()
    ax.set_prop_cycle(AGILITUS_COLORS)

    for hoist_class in HoistClass:
        phi_2_values = [
            phi_2(v_h=v, hoist_class=hoist_class, drive_class=drive_class) for v in v_h
        ]
        plt.plot(v_h, phi_2_values, label=hoist_class)

    plt.xlabel("Hoist speed $v_h$ (m/s)")
    plt.ylabel(r"Dynamic hoisting factor, $\phi_2$")
    plt.title(f"Dynamic hoisting factor for drive class {drive_class}")
    plt.legend()
    plt.show()
