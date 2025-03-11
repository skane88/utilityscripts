"""
Implements desing of a concrete slab to CCAA T48
"""

from enum import StrEnum
from math import log10
from pathlib import Path

import numpy as np
import polars as pl

_DATA_PATH = Path(Path(__file__).parent) / Path("data")


class LoadingType(StrEnum):
    """
    Types of loading from CCAA
    """

    WHEEL = "wheel"
    POINT = "point"
    DISTRIBUTED = "distributed"


class MaterialFactor(StrEnum):
    """
    Defines 3x cases for selecting the material factor k_1, based on
    the guidance in CCAA T48
    """

    CONSERVATIVE = "conservative"
    MIDRANGE = "midrange"
    UNCONSERVATIVE = "unconservative"


MATERIAL_FACTOR = pl.DataFrame(
    {
        "load_type": [LoadingType.WHEEL, LoadingType.POINT, LoadingType.DISTRIBUTED],
        MaterialFactor.CONSERVATIVE: [0.85, 0.75, 0.75],
        MaterialFactor.UNCONSERVATIVE: [0.95, 0.85, 0.85],
    }
)


def k_1(*, loading_type: LoadingType, material_factor: MaterialFactor) -> float:
    """
    Calculates the material factor k_1, based on the loading type and material factor

    Parameters
    ----------
    loading_type : LoadingType
        The type of loading
    material_factor : MaterialFactor
        The material factor

    Returns
    -------
    float
        The material factor k_1
    """

    if material_factor == MaterialFactor.MIDRANGE:
        return (
            MATERIAL_FACTOR.filter(pl.col("load_type") == loading_type)
            .with_columns(
                pl.mean_horizontal(
                    [MaterialFactor.CONSERVATIVE, MaterialFactor.UNCONSERVATIVE]
                ).alias(MaterialFactor.MIDRANGE)
            )[MaterialFactor.MIDRANGE]
            .item()
        )

    return MATERIAL_FACTOR.filter(pl.col("load_type") == loading_type)[
        material_factor
    ].item()


def k_2(*, no_cycles: float, load_type: LoadingType = LoadingType.WHEEL) -> float:
    """
    Calculates the load repetition factor k_2, based on the number of cycles

    Notes
    -----
    CCAA suggests that the equation used is only valid for greater than 50 cycles, and
    that at a no_cycles = 1.0 then k_2 = 1.00. It has been decided to use this euquation
    in the range of 1->50 cycles anyway because at 1 cycle k_2 = 0.98, slightly less
    than CCAA's recommended value of 1.00.

    Parameters
    ----------
    no_cycles : float
        The number of cycles
    load_type : LoadingType
        The type of loading. If LoadingType.DISTRIBUTED, then k_2 = 0.75

    Returns
    -------
    float
        The load repetition factor k_2
    """

    if load_type == LoadingType.DISTRIBUTED:
        return 0.75

    if no_cycles <= 1.0:
        return 1.00

    return max(0.73 - 0.0846 * (log10(no_cycles) - 3), 0.50)


def f_all(*, k_1, k_2, f_cf):
    """
    Calculates the allowable stress in the concrete.

    Parameters
    ----------
    k_1 : float
        The material factor k_1
    k_2 : float
        The load repetition factor k_2
    f_cf : float
        The characteristic flexural strength of the concrete.

    Returns
    -------
    float
        The allowable concrete stress.
    """

    return k_1 * k_2 * f_cf


def w_fi(
    *,
    depth: float,
    normalising_length: float,
    loading_type: LoadingType,
) -> float:
    """
    Calculates the layer weighting factor w_fi.

    Parameters
    ----------
    depth : float
        The depth of the layer
    normalising_length : float
        The normalising length. Depends on the loading type - refer to CCAA T48.
    loading_type : LoadingType
        The type of loading

    Returns
    -------
    float
        The layer weighting factor w_fi
    """

    data_excel = _DATA_PATH / Path("ccaa_t48_data.xlsx")

    if loading_type == LoadingType.WHEEL:
        data = pl.read_excel(data_excel, sheet_name="w_fi_wheels")

    elif loading_type == LoadingType.POINT:
        data = pl.read_excel(data_excel, sheet_name="w_fi_posts")

    elif loading_type == LoadingType.DISTRIBUTED:
        data = pl.read_excel(data_excel, sheet_name="w_fi_distributed")

    data = data.sort("relative_depth")

    relative_depth = depth / normalising_length

    depth_p = data["relative_depth"].to_numpy()
    w_fi_p = data["w_fi"].to_numpy()

    return np.interp(relative_depth, depth_p, w_fi_p)


def e_se(
    h_layers: list[float],
    e_layers: list[float],
    normalising_length: float,
    loading_type: LoadingType,
) -> float:
    """
    Calculate the equivalent stiffness of a system of soil layers.
    """

    h_layers = np.asarray(h_layers)
    e_layers = np.asarray(e_layers)
    z = np.cumsum(h_layers) - 0.5 * h_layers

    w_fi_layers = np.asarray(
        [
            w_fi(
                depth=z[i],
                normalising_length=normalising_length,
                loading_type=loading_type,
            )
            for i in range(len(h_layers))
        ]
    )

    return np.sum(w_fi_layers * h_layers) / np.sum(w_fi_layers * h_layers / e_layers)
