"""
Implements desing of a concrete slab to CCAA T48
"""

from enum import StrEnum
from math import log10
from pathlib import Path

import matplotlib.pyplot as plt
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


class LoadLocation(StrEnum):
    """
    Where is the load located?
    """

    INTERNAL = "internal"
    EDGE = "edge"


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


def plot_w_fi():
    data_wheel = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="w_fi_wheels"
    )
    data_point = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="w_fi_posts"
    )
    data_distributed = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="w_fi_distributed"
    )

    fig, ax = plt.subplots()

    ax.plot(
        data_wheel["w_fi"], data_wheel["relative_depth"], label="Wheel Loading (X=S)"
    )
    ax.plot(
        data_point["w_fi"],
        data_point["relative_depth"],
        label="Point Loading (X=f(x, y))",
    )
    ax.plot(
        data_distributed["w_fi"],
        data_distributed["relative_depth"],
        label="Distributed Loading (X=W)",
    )

    ax.set_xlabel("Weight Factor W_fi")
    ax.set_ylabel("Relative Depth (z / X)")
    ax.legend()
    ax.set_xlim(0, 1.0)
    ax.set_ylim(12, 0)
    ax.grid(visible=True)

    plt.show()


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


def e_ss_from_e_sl(e_sl: float, b: float) -> float:
    """
    Calculate the short term Young's modulus of the soil from the long term value.
    """

    return e_sl / b


def e_sl_from_cbr(cbr: float) -> float:
    """
    Calculate the long term Young's modulus of the soil from the CBR.

    Uses data interpolated from CCAA T48 Figure 1.24.

    Parameters
    ----------
    cbr : float
        The CBR value

    Returns
    -------
    float
        The long term Young's modulus of the soil
    """

    data = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="e_sl_from_cbr"
    )

    cbr_vals = data["cbr"].to_numpy()
    e_sl_vals = data["e_sl"].to_numpy()

    if cbr < data["cbr"].min():
        raise ValueError(
            f"CBR value of {cbr} is less than "
            + f"the minimum value of {data['cbr'].min():.1f}"
        )

    if cbr > data["cbr"].max():
        raise ValueError(
            f"CBR value of {cbr} is greater than "
            + f"the maximum value of {data['cbr'].max():.0f}"
        )

    return np.interp(cbr, cbr_vals, e_sl_vals)


def k_3(load_location: LoadLocation) -> float:
    """
    Calculate the calibration factor for geotechnical behaviour, k_3 as per CCAA T48.
    """

    mapping = {LoadLocation.INTERNAL: 1.2, LoadLocation.EDGE: 1.05}

    return mapping[load_location]


def k_4(f_c: float) -> float:
    """
    Calculate the calibration factor for concrete strength, k_4 as per CCAA T48.
    """

    data = pl.read_excel(_DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="k_4")

    f_c_vals = data["f_c"].to_numpy()
    k_4_vals = data["k_4"].to_numpy()

    if f_c < data["f_c"].min():
        raise ValueError(
            f"f_c value of {f_c} is less than "
            + f"the minimum value of {data['f_c'].min():.0f}"
        )

    if f_c > data["f_c"].max():
        raise ValueError(
            f"f_c value of {f_c} is greater than "
            + f"the maximum value of {data['f_c'].max():.0f}"
        )

    return np.interp(f_c, f_c_vals, k_4_vals)


def f_e1(e_ss: float, load_type: LoadingType, load_location: LoadLocation) -> float:
    """
    Calculate the short term Young's modulus factor, f_e1 as per CCAA T48 section 3.3.8
    """

    if load_type == LoadingType.WHEEL and load_location == LoadLocation.INTERNAL:
        data = pl.read_excel(
            _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht11_f_e1"
        )
    elif load_type == LoadingType.WHEEL and load_location == LoadLocation.EDGE:
        data = pl.read_excel(
            _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht12_f_e1"
        )
    elif load_type == LoadingType.POINT:
        data = pl.read_excel(
            _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht13_f_e1"
        )
    elif load_type == LoadingType.DISTRIBUTED:
        data = pl.read_excel(
            _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht14_f_e1"
        )
    else:
        raise ValueError(
            f"Invalid load type: {load_type} or load location: {load_location}"
        )

    e_ss_vals = data["e_ss"].to_numpy()
    f_e1_vals = data["f_e1"].to_numpy()

    if e_ss < data["e_ss"].min():
        raise ValueError(
            f"e_ss value of {e_ss} is less than "
            + f"the minimum value of {data['e_ss'].min():.0f}"
        )

    if e_ss > data["e_ss"].max():
        raise ValueError(
            f"e_ss value of {e_ss} is greater than "
            + f"the maximum value of {data['e_ss'].max():.0f}"
        )

    return np.interp(e_ss, e_ss_vals, f_e1_vals)


def f_1(*, f_all, f_e1, f_h1, f_s1, k_3, k_4) -> float:
    """
    Calculate the equvialent stress factor, F_1 as per CCAA T48 section 3.3.8

    Parameters
    ----------
    f_all : float
        The allowable stress in the concrete
    f_e1 : float
        The short term Young's modulus factor.
    f_h1 : float
        The depth of soil factor.
    f_s1 : float
        The wheel spacing factor.
    k_3 : float
        A calibration factor for geotechnical behaviour.
    k_4 : float
        A calibration factor for concrete strength.

    Returns
    -------
    float
        The equivalent stress factor, F_1
    """

    return f_all * f_e1 * f_h1 * f_s1 * k_3 * k_4
