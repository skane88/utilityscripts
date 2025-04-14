"""
File to contain functions for working with crane loads to the AS522x series of
standards.
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from utilityscripts.plotting import AGILITUS_COLORS

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
