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


def s6_1_2_1_hoist_class(*, delta: float) -> HoistClass:
    """
    Get the hoist class. The hoist class is assigned based on the elastic stiffness
    of the crane. This is approximated by the characteristic deflection, delta.

    Parameters
    ----------
    delta : float
        The characteristic deflection. In m.

    Returns
    -------
    HoistClass
        The hoist class.
    """

    beta_2_data = _get_beta_2_data()
    beta_2_data = beta_2_data.filter(pl.col("delta_max") > delta)
    beta_2_data = beta_2_data.filter(pl.col("delta_min") <= delta)

    return HoistClass(beta_2_data["hoist_class"][0])


def s6_1_2_1_beta_2(*, hoist_class: HoistClass):
    """
    Get the value of beta_2 for a given hoisting class.

    Parameters
    ----------
    hoist_class : HoistClass
        The hoist class.

    Returns
    -------
    float
        The value of beta_2. In s/m.
    """

    beta_2_data = _get_beta_2_data()
    beta_2_data = beta_2_data.filter(pl.col("hoist_class") == hoist_class)
    return beta_2_data["beta_2"][0]


def s6_1_2_1_phi_2_min(*, hoist_class: HoistClass, drive_class: DriveClass):
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


def s6_1_2_1_phi_2(*, v_h: float, hoist_class: HoistClass, drive_class: DriveClass):
    """
    Calculate the dynamic hoisting factor, phi_2, as per S6.1.2.1 of AS5221.1-2021.

    Parameters
    ----------
    v_h : float
        The hoist speed. In m/s.
    hoist_class : HoistClass
        The hoist class.
    drive_class : DriveClass
        The drive class.

    Returns
    -------
    float
        The dynamic hoisting factor.
    """

    beta_2 = s6_1_2_1_beta_2(hoist_class=hoist_class)
    phi_2_min = s6_1_2_1_phi_2_min(hoist_class=hoist_class, drive_class=drive_class)

    return phi_2_min + beta_2 * v_h


def plot_phi_2(drive_class: DriveClass):
    v_h = np.linspace(0, 2, 100)

    ax = plt.gca()
    ax.set_prop_cycle(AGILITUS_COLORS)

    for hoist_class in HoistClass:
        phi_2_values = [
            s6_1_2_1_phi_2(v_h=v, hoist_class=hoist_class, drive_class=drive_class)
            for v in v_h
        ]
        plt.plot(v_h, phi_2_values, label=hoist_class)

    plt.xlabel("Hoist speed $v_h$ (m/s)")
    plt.ylabel(r"Dynamic hoisting factor, $\phi_2$")
    plt.title(f"Dynamic hoisting factor for drive class {drive_class}")
    plt.legend()
    plt.show()


def s5_2_a_h(*, m_h: float) -> float:
    """
    Calculate a nominal area for a hoisted load, for use in wind load calcs.

    Parameters
    ----------
    m_h : float
        The mass of the hoisted load. In kg.

    Returns
    -------
    float
        The nominal area. In m².
    """

    return 0.0005 * m_h


def s5_2_f_h(*, q_z: float, a_h: float, c_h: float = 2.40) -> float:
    """
    Calculate the wind load on a hoisted load.

    Parameters
    ----------
    q_z : float
        The design wind pressure at the reference height. In Pa.
        Typically the reference height is the highest point of the crane during the
        lift.
    a_h : float
        The nominal area of the hoisted load. In m².
    c_h : float, optional
        The wind drag coefficient for the hoisted load. Default is 2.40.

    Returns
    -------
    float
        The wind load on the hoisted load. In N.
    """

    return q_z * c_h * a_h
