"""
Implements desing of a concrete slab to CCAA T48
"""

from copy import deepcopy
from enum import StrEnum
from functools import lru_cache
from math import log10
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import ticker
from scipy.interpolate import CloughTocher2DInterpolator  # type: ignore

from utilityscripts.plotting import AGILITUS_COLORS

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


class LoadDuration(StrEnum):
    """
    What is the load duration
    """

    SHORT = "short"
    LONG = "long"


MATERIAL_FACTOR = pl.DataFrame(
    {
        "load_type": [LoadingType.WHEEL, LoadingType.POINT, LoadingType.DISTRIBUTED],
        MaterialFactor.CONSERVATIVE: [0.85, 0.75, 0.75],
        MaterialFactor.UNCONSERVATIVE: [0.95, 0.85, 0.85],
    }
)


class Soil:
    """
    Represents a soil.
    """

    def __init__(self, *, e_sl: float, e_ss: float, soil_name: str | None = None):
        """
        Create a Soil object.

        Parameters
        ----------
        e_sl : float
            The long term Young's modulus of the soil, in MPa.
        e_ss : float
            The short term Young's modulus of the soil, in MPa.
        soil_name : str | None
            An optional name to give the soil.
        """

        self._e_sl = e_sl
        self._e_ss = e_ss
        self._soil_name = soil_name

    @property
    def e_sl(self) -> float:
        """
        Get the long term Young's modulus of the soil. In MPa.
        """

        return self._e_sl

    @property
    def e_ss(self) -> float:
        """
        Get the short term Young's modulus of the soil. In MPa.
        """

        return self._e_ss

    @property
    def soil_name(self) -> str | None:
        """
        An optional name for the soil type.
        """

        return self._soil_name

    def __repr__(self):
        return (
            f"{type(self).__name__}: "
            + f"e_sl = {self.e_sl:.1f} MPa, "
            + f"e_ss = {self.e_ss:.1f} MPa"
            + f"{', ' + self.soil_name if self.soil_name else '.'}"
        )


class SoilProfile:
    """
    Represents a soil profile.
    """

    def __init__(self, *, h_layers: list[float], soils: list[Soil]):
        """
        Create a SoilProfile object.

        Parameters
        ----------
        h_layers : list[float]
            The thickness of each soil layer. List from the shallowest to the deepest.
        soils : list[Soil]
            The soils in the profile
        """

        self._h_layers = h_layers
        self._soils = soils

    @property
    def h_layers(self) -> list[float]:
        return self._h_layers

    @property
    def soils(self) -> list[Soil]:
        return self._soils

    @property
    def h_total(self) -> float:
        return sum(self.h_layers)

    def e_ss(self, *, normalising_length: float, loading_type: LoadingType) -> float:
        if len(self.soils) == 1:
            return self.soils[0].e_ss

        return e_se(
            h_layers=self.h_layers,
            e_layers=[soil.e_ss for soil in self.soils],
            normalising_length=normalising_length,
            loading_type=loading_type,
        )

    def e_sl(self, *, normalising_length: float, loading_type: LoadingType) -> float:
        if len(self.soils) == 1:
            return self.soils[0].e_sl

        return e_se(
            h_layers=self.h_layers,
            e_layers=[soil.e_sl for soil in self.soils],
            normalising_length=normalising_length,
            loading_type=loading_type,
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}: "
            + f"no. layers = {len(self.h_layers)}, "
            + f"total thickness = {self.h_total:.1f}."
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


@lru_cache(maxsize=None)
def _get_w_fi_data() -> dict[LoadingType, pl.DataFrame]:
    """
    Get the w_fi data into a dictionary for easy use later.

    Notes
    -----
    Method cached to eliminate unneccsary Excel reads.
    """

    data_excel = _DATA_PATH / Path("ccaa_t48_data.xlsx")

    data_wheel = pl.read_excel(data_excel, sheet_name="w_fi_wheels")
    data_point = pl.read_excel(data_excel, sheet_name="w_fi_posts")
    data_distributed = pl.read_excel(data_excel, sheet_name="w_fi_distributed")

    return {
        LoadingType.WHEEL: data_wheel,
        LoadingType.POINT: data_point,
        LoadingType.DISTRIBUTED: data_distributed,
    }


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

    data = _get_w_fi_data()[loading_type]
    data = data.sort("relative_depth")

    relative_depth = depth / normalising_length

    return np.interp(relative_depth, data["relative_depth"], data["w_fi"])


def plot_w_fi():
    """
    Plot the w_fi curves.

    Primarily useful for debugging.
    """

    data = _get_w_fi_data()

    fig, ax = plt.subplots()

    ax.set_prop_cycle(AGILITUS_COLORS)

    ax.plot(
        data[LoadingType.WHEEL]["w_fi"],
        data[LoadingType.WHEEL]["relative_depth"],
        label="Wheel Loading (X=S)",
    )
    ax.plot(
        data[LoadingType.POINT]["w_fi"],
        data[LoadingType.POINT]["relative_depth"],
        label="Point Loading (X=f(x, y))",
    )
    ax.plot(
        data[LoadingType.DISTRIBUTED]["w_fi"],
        data[LoadingType.DISTRIBUTED]["relative_depth"],
        label="Distributed Loading (X=W)",
    )

    ax.set_xlabel("Weight Factor W_fi")
    ax.set_ylabel("Relative Depth (z / X)")
    ax.legend()
    ax.set_xlim(0, 1.0)
    ax.set_ylim(12, 0)
    ax.grid(visible=True)

    ax.set_title("Layer Weighting Factors for Soil Modulus")

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


@lru_cache(maxsize=None)
def _e_sl_data() -> pl.DataFrame:
    """
    Get the e_sl data into a DataFrame for easy use later.

    Notes
    -----
    Method cached to eliminate unneccsary Excel reads.
    """

    return pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="e_sl_from_cbr"
    )


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

    data = _e_sl_data()

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

    return np.interp(cbr, data["cbr"], data["e_sl"])


def plot_e_sl_from_cbr():
    """
    Plot the e_sl from cbr data.

    Primarily useful for debugging.
    """

    data = _e_sl_data()

    fig, ax = plt.subplots()

    ax.set_prop_cycle(AGILITUS_COLORS)

    ax.plot(data["cbr"], data["e_sl"], label="E_sl from CBR")

    ax.set_xlabel("CBR")
    ax.set_ylabel("E_sl")

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100])  # ticks to match T48
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}"))
    ax.set_xlim(1, 100)
    ax.grid(visible=True)

    ax.set_xlabel("CBR (%)")
    ax.set_ylabel("E_sl (MPa)")

    ax.set_title("E_sl to CBR Correlation")

    ax.legend()
    plt.show()


def k_3(load_location: LoadLocation) -> float:
    """
    Calculate the calibration factor for geotechnical behaviour, k_3 as per CCAA T48.
    """

    mapping = {LoadLocation.INTERNAL: 1.2, LoadLocation.EDGE: 1.05}

    return mapping[load_location]


@lru_cache(maxsize=None)
def _k_4_data() -> pl.DataFrame:
    """
    Get the k_4 data into a DataFrame for easy use later.

    Notes
    -----
    Method cached to eliminate unneccsary Excel reads.
    """

    return pl.read_excel(_DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="k_4")


def k_4(f_c: float) -> float:
    """
    Calculate the calibration factor for concrete strength, k_4 as per CCAA T48.
    """

    data = _k_4_data()

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

    return np.interp(f_c, data["f_c"], data["k_4"])


@lru_cache(maxsize=None)
def _f_e_data() -> dict[LoadingType, dict[LoadLocation, pl.DataFrame]]:
    """
    Get the f_e1 data into a dictionary for easy use later.

    Notes
    -----
    Method cached to eliminate unneccesary Excel reads.
    """

    data = {LoadingType.WHEEL: {}, LoadingType.POINT: {}, LoadingType.DISTRIBUTED: {}}

    data[LoadingType.WHEEL][LoadLocation.INTERNAL] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht11_f_e"
    )
    data[LoadingType.WHEEL][LoadLocation.EDGE] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht12_f_e"
    )
    data[LoadingType.POINT][LoadLocation.INTERNAL] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht13_f_e_interior"
    )
    data[LoadingType.POINT][LoadLocation.EDGE] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht13_f_e_edge"
    )
    data[LoadingType.DISTRIBUTED][LoadLocation.INTERNAL] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht14_f_e"
    )
    data[LoadingType.DISTRIBUTED][LoadLocation.EDGE] = deepcopy(
        data[LoadingType.DISTRIBUTED][LoadLocation.INTERNAL]
    )

    return data


def f_e(e_sx: float, load_type: LoadingType, load_location: LoadLocation) -> float:
    """
    Calculate the Young's modulus factor, f_e as per CCAA T48 section 3.3.8

    Parameters
    ----------
    e_sx : float
        The Young's modulus of the soil. Use the short term Young's modulus for short
        duration loads, the long term young's modulus for long term loads.
    load_type : LoadingType
        The type of loading
    load_location : LoadLocation
        The location of the load

    Returns
    -------
    float
        The short term Young's modulus factor, f_e
    """

    data = _f_e_data()[load_type][load_location]

    if e_sx < data["e_sx"].min():
        raise ValueError(
            f"e_ss value of {e_sx} is less than "
            + f"the minimum value of {data['e_sx'].min():.0f}"
        )

    if e_sx > data["e_sx"].max():
        raise ValueError(
            f"e_sx value of {e_sx} is greater than "
            + f"the maximum value of {data['e_sx'].max():.0f}"
        )

    return np.interp(e_sx, data["e_sx"], data["f_e"])


def plot_f_e():
    """
    Plot the f_e data.

    Primarily useful for debugging.
    """

    data = _f_e_data()

    fig, ax = plt.subplots()

    ax.set_prop_cycle(AGILITUS_COLORS)

    ax.plot(
        data[LoadingType.WHEEL][LoadLocation.INTERNAL]["e_ss"],
        data[LoadingType.WHEEL][LoadLocation.INTERNAL]["f_e"],
        label="Wheel Loading (Interior) (X=S)",
    )
    ax.plot(
        data[LoadingType.WHEEL][LoadLocation.EDGE]["e_ss"],
        data[LoadingType.WHEEL][LoadLocation.EDGE]["f_e"],
        label="Wheel Loading (Edge) (X=S)",
    )
    ax.plot(
        data[LoadingType.POINT][LoadLocation.INTERNAL]["e_ss"],
        data[LoadingType.POINT][LoadLocation.INTERNAL]["f_e"],
        label="Point Loading (Interior) (X=f(x, y))",
    )
    ax.plot(
        data[LoadingType.POINT][LoadLocation.EDGE]["e_ss"],
        data[LoadingType.POINT][LoadLocation.EDGE]["f_e"],
        label="Point Loading (Edge) (X=f(x, y))",
    )
    ax.plot(
        data[LoadingType.DISTRIBUTED][LoadLocation.INTERNAL]["e_ss"],
        data[LoadingType.DISTRIBUTED][LoadLocation.INTERNAL]["f_e"],
        label="Distributed Loading (Interior) (X=W)",
    )
    ax.plot(
        data[LoadingType.DISTRIBUTED][LoadLocation.EDGE]["e_ss"],
        data[LoadingType.DISTRIBUTED][LoadLocation.EDGE]["f_e"],
        label="Distributed Loading (Edge) (X=W)",
    )

    ax.set_xlabel("E_ss")
    ax.set_ylabel("F_E")
    ax.legend()
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 5)
    ax.grid(visible=True)

    ax.set_title("Factor F_E to E_ss")

    plt.show()


@lru_cache(maxsize=None)
def _f_s_data() -> dict[LoadingType, dict[LoadLocation, pl.DataFrame]]:
    """
    Get the f_s data into a dictionary for easy use later.

    Notes
    -----
    Method cached to eliminate unneccesary Excel reads.
    """

    data = {LoadingType.WHEEL: {}, LoadingType.POINT: {}, LoadingType.DISTRIBUTED: {}}

    data[LoadingType.WHEEL][LoadLocation.INTERNAL] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht11_f_s"
    )
    data[LoadingType.WHEEL][LoadLocation.EDGE] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht12_f_s"
    )
    data[LoadingType.POINT][LoadLocation.INTERNAL] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht13_f_s"
    )
    data[LoadingType.POINT][LoadLocation.EDGE] = deepcopy(
        data[LoadingType.POINT][LoadLocation.INTERNAL]
    )
    data[LoadingType.DISTRIBUTED][LoadLocation.INTERNAL] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht14_f_s"
    )
    data[LoadingType.DISTRIBUTED][LoadLocation.EDGE] = deepcopy(
        data[LoadingType.DISTRIBUTED][LoadLocation.INTERNAL]
    )

    return data


def f_s(x: float, load_type: LoadingType, load_location: LoadLocation) -> float:
    """
    Calculate the spacing factor, f_s as per CCAA T48 section 3.3.8

    Parameters
    ----------
    x : float
        The spacing parameter.

        x = S for wheel loads,
        x = average post spacing for post loads.
        x = W for distributed loads.

    load_type : LoadingType
        The type of loading
    load_location : LoadLocation
        The location of the load

    Returns
    -------
    float
        The spacing factor, f_s
    """

    data = _f_s_data()[load_type][load_location]

    if x < data["x"].min():
        raise ValueError(
            f"x value of {x} is less than "
            + f"the minimum value of {data['x'].min():.0f}"
        )

    if x > data["x"].max():
        raise ValueError(
            f"x value of {x} is greater than "
            + f"the maximum value of {data['x'].max():.0f}"
        )

    return np.interp(x, data["x"], data["f_s"])


def plot_f_s():
    """
    Plot the f_s data.

    Primarily useful for debugging.
    """

    data = _f_s_data()

    fig, ax = plt.subplots()

    ax.set_prop_cycle(AGILITUS_COLORS)

    ax.plot(
        data[LoadingType.WHEEL][LoadLocation.INTERNAL]["x"],
        data[LoadingType.WHEEL][LoadLocation.INTERNAL]["f_s"],
        label="Wheel Loading (Interior) (X=S)",
    )
    ax.plot(
        data[LoadingType.WHEEL][LoadLocation.EDGE]["x"],
        data[LoadingType.WHEEL][LoadLocation.EDGE]["f_s"],
        label="Wheel Loading (Edge) (X=S)",
    )
    ax.plot(
        data[LoadingType.POINT][LoadLocation.INTERNAL]["x"],
        data[LoadingType.POINT][LoadLocation.INTERNAL]["f_s"],
        label="Point Loading (Interior) (X=f(x, y))",
    )
    ax.plot(
        data[LoadingType.POINT][LoadLocation.EDGE]["x"],
        data[LoadingType.POINT][LoadLocation.EDGE]["f_s"],
        label="Point Loading (Edge) (X=f(x, y))",
    )
    ax.plot(
        data[LoadingType.DISTRIBUTED][LoadLocation.INTERNAL]["x"],
        data[LoadingType.DISTRIBUTED][LoadLocation.INTERNAL]["f_s"],
        label="Distributed Loading (Interior) (X=W)",
    )
    ax.plot(
        data[LoadingType.DISTRIBUTED][LoadLocation.EDGE]["x"],
        data[LoadingType.DISTRIBUTED][LoadLocation.EDGE]["f_s"],
        label="Distributed Loading (Edge) (X=W)",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("F_S")
    ax.legend()
    ax.set_xlim(1, 5)
    ax.set_ylim(0.5, 2.0)
    ax.grid(visible=True)

    ax.set_title("Factor F_S to x")

    plt.show()


@lru_cache(maxsize=None)
def _f_h_data() -> dict[LoadingType, dict[LoadLocation, pl.DataFrame]]:
    """
    Get the f_h data into a dictionary for easy use later.

    Notes
    -----
    Method cached to eliminate unneccesary Excel reads.
    """

    data = {LoadingType.WHEEL: {}, LoadingType.POINT: {}, LoadingType.DISTRIBUTED: {}}

    data[LoadingType.WHEEL][LoadLocation.INTERNAL] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht11_f_h"
    )
    data[LoadingType.WHEEL][LoadLocation.EDGE] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht12_f_h"
    )
    data[LoadingType.POINT][LoadLocation.INTERNAL] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht13_f_h"
    )
    data[LoadingType.POINT][LoadLocation.EDGE] = deepcopy(
        data[LoadingType.POINT][LoadLocation.INTERNAL]
    )
    data[LoadingType.DISTRIBUTED][LoadLocation.INTERNAL] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht14_f_h"
    )
    data[LoadingType.DISTRIBUTED][LoadLocation.EDGE] = deepcopy(
        data[LoadingType.DISTRIBUTED][LoadLocation.INTERNAL]
    )

    return data


def f_h(*, h: float, load_type: LoadingType, load_location: LoadLocation) -> float:
    """
    Calculate the soil thickness factor, f_h as per CCAA T48 section 3.3.8

    Parameters
    ----------
    h : float
        The soil thickness, in m.
    load_type : LoadingType
        The type of loading
    load_location : LoadLocation
        The location of the load

    Returns
    -------
    float
        The soil thickness factor, f_h
    """

    data = _f_h_data()[load_type][load_location]

    if h < data["h"].min():
        raise ValueError(
            f"h value of {h} is less than "
            + f"the minimum value of {data['h'].min():.0f}"
        )

    if h > data["h"].max():
        raise ValueError(
            f"h value of {h} is greater than "
            + f"the maximum value of {data['h'].max():.0f}"
        )

    return np.interp(h, data["h"], data["f_h"])


def plot_f_h():
    """
    Plot the f_h data.

    Primarily useful for debugging.
    """

    data = _f_h_data()

    fig, ax = plt.subplots()

    ax.set_prop_cycle(AGILITUS_COLORS)

    ax.plot(
        data[LoadingType.WHEEL][LoadLocation.INTERNAL]["h"],
        data[LoadingType.WHEEL][LoadLocation.INTERNAL]["f_h"],
        label="Wheel Loading (Interior)",
    )
    ax.plot(
        data[LoadingType.WHEEL][LoadLocation.EDGE]["h"],
        data[LoadingType.WHEEL][LoadLocation.EDGE]["f_h"],
        label="Wheel Loading (Edge)",
    )
    ax.plot(
        data[LoadingType.POINT][LoadLocation.INTERNAL]["h"],
        data[LoadingType.POINT][LoadLocation.INTERNAL]["f_h"],
        label="Point Loading (Interior)",
    )
    ax.plot(
        data[LoadingType.POINT][LoadLocation.EDGE]["h"],
        data[LoadingType.POINT][LoadLocation.EDGE]["f_h"],
        label="Point Loading (Edge)",
    )
    ax.plot(
        data[LoadingType.DISTRIBUTED][LoadLocation.INTERNAL]["h"],
        data[LoadingType.DISTRIBUTED][LoadLocation.INTERNAL]["f_h"],
        label="Distributed Loading (Interior)",
    )
    ax.plot(
        data[LoadingType.DISTRIBUTED][LoadLocation.EDGE]["h"],
        data[LoadingType.DISTRIBUTED][LoadLocation.EDGE]["f_h"],
        label="Distributed Loading (Edge)",
    )

    ax.set_xlabel("h (m)")
    ax.set_ylabel("F_H")
    ax.legend()
    ax.set_xlim(0, 16)
    ax.set_ylim(0.5, 3)
    ax.grid(visible=True)

    ax.set_title("Factor F_H vs H")

    plt.show()


def f_12(
    *,
    f_all: float,
    f_e12: float,
    f_h12: float,
    f_s12: float,
    k_3: float,
    k_4: float,
) -> float:
    """
    Calculate the equvialent stress factor, F_1 or F2 as per CCAA T48 section 3.3.8

    Parameters
    ----------
    f_all : float
        The allowable stress in the concrete, in MPa
    f_e12 : float
        The short term Young's modulus factor F_E1 or F_E2.
    f_h12 : float
        The depth of soil factor F_H1 or F_H2.
    f_s12 : float
        The wheel spacing factor F_S1 or F_S2.
    k_3 : float
        A calibration factor for geotechnical behaviour.
    k_4 : float
        A calibration factor for concrete strength.

    Returns
    -------
    float
        The equivalent stress factor, F_1 or F_2
    """

    return f_all * f_e12 * f_h12 * f_s12 * k_3 * k_4


def f_3(
    *,
    p: float,
    f_all: float,
    f_e3: float,
    f_h3: float,
    f_s3: float,
) -> float:
    """
    Calculate the equvialent stress factor, F_3 as per CCAA T48 section 3.3.8

    Parameters
    ----------
    p : float
        The load, in kN
    f_all : float
        The allowable stress in the concrete, in MPa
    f_e3 : float
        The short term Young's modulus factor F_E3.
    f_h3 : float
        The depth of soil factor F_H3.
    f_s3 : float
        The wheel spacing factor F_S3.

    Returns
    -------
    float
        The equivalent stress factor, F_3
    """

    return (1000 / p) * f_all * f_e3 * f_h3 * f_s3


def f_4(
    *,
    p: float,
    f_all: float,
    f_e4: float,
    f_h4: float,
    f_s4: float,
) -> float:
    """
    Calculate the equvialent stress factor, F_4 as per CCAA T48 section 3.3.8

    Parameters
    ----------
    load_location : LoadLocation
        The location of the load
    P : float
        The distributed load, in kPa
    f_all : float
        The allowable stress in the concrete, in MPa
    f_e4 : float
        The short term Young's modulus factor F_E4.
    f_h4 : float
        The depth of soil factor F_H4.
    f_s4 : float
        The wheel spacing factor F_S4.

    Returns
    -------
    float
        The equivalent stress factor, F_4
    """

    return (1000 / p) * f_all * f_e4 * f_h4 * f_s4


@lru_cache(maxsize=None)
def _t_12_data() -> dict[LoadLocation, pl.DataFrame]:
    """
    Get the data for thicknesses based on F_1 and F_2 into a DataFrame for easy use
    later.

    Notes
    -----
    - The data is interpolated from CCAA T48 Chart 1.1 & 1.2.
    - A separate method is used so that the call into the spreadsheet can be cached.
    """

    data = {}

    data[LoadLocation.INTERNAL] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht11_thickness"
    )
    data[LoadLocation.EDGE] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht12_thickness"
    )

    return data


@lru_cache(maxsize=None)
def _t_12_interp(load_location: LoadLocation) -> CloughTocher2DInterpolator:
    """
    Get the interpolator for the t_12 data.

    Notes
    -----
    - The Clough - Tocher interpolator is used because the Delaunay triangulation used
      introduces some errors in the Linear interpolation, resulting in very local
      significant underestimates of thickness. The Clough-Tocher interpolation does
      result in some minor overestimates the required thickness compared to
      a (correct) linear interpolation. This should be conservative.
    - The interpolator is cached to eliminate re-building the triangulation every call.
    """

    data = _t_12_data()[load_location]

    return CloughTocher2DInterpolator(
        data[["f", "p"]],
        data["t"],
    )


def t_12(
    *, f_12: float | np.ndarray, p: float | np.ndarray, load_location: LoadLocation
) -> float | np.ndarray:
    """
    Calculate the required slab thickness for wheel loads.

    Notes
    -----
    - A Clough - Tocher interpolation is used because the Delaunay triangulation required
      for a 2D interpolation introduces some errors in the Linear interpolation.
      This results in very local significant underestimates of thickness when using
      a linear interpolator. The Clough-Tocher interpolation does
      result in some minor overestimates the required thickness compared to
      a (correct) linear interpolation. This should be conservative.

    Parameters
    ----------
    f_12 : float | np.ndarray
        The equivalent stress factor, F_1 or F_2. If a numpy array is passed in then
        multiple thicknesses will be calculated at once.
    p : float | np.ndarray
        The load, in kN. If a numpy array is passed in then multiple thicknesses will
        be calculated at once.
    load_location : LoadLocation
        The location of the load

    Returns
    -------
    float | np.ndarray
        The required slab thickness, in mm
    """

    return _t_12_interp(load_location)(f_12, p)


def plot_t_12_data(load_location: LoadLocation):
    """
    Plot the t_12 data.

    Primarily useful for debugging.
    """

    data = _t_12_data()[load_location]

    fig, ax = plt.subplots()

    ax.set_prop_cycle(AGILITUS_COLORS)

    for p in data["p"].unique():
        line_data = data.filter(pl.col("p") == p)
        ax.plot(line_data["f"], line_data["t"], label=f"p = {p} kN")
        ax.scatter(line_data["f"], line_data["t"])

    ax.set_xlabel("F_12")
    ax.set_ylabel("Thickness (mm)")
    ax.legend()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 600)
    ax.grid(visible=True)
    ax.axhspan(400, 600, color="grey", alpha=0.34)
    ax.text(
        x=3,
        y=500,
        s="Use computer analysis\nin shaded area.",
        bbox={"facecolor": "lightgrey", "edgecolor": "lightgrey"},
        fontsize=8,
    )

    plt.show()


def plot_t_12_space(load_location: LoadLocation):
    """
    Plot the solution space available for T_1 & T_2

    Primarily useful for debugging.
    """

    data = _t_12_data()[load_location]

    f_space = np.linspace(0, 5, 100)
    p_space = np.linspace(0, 100, 50)
    p_space = np.hstack((p_space, np.linspace(0, 200, 25)))
    p_space = np.hstack(
        (
            p_space,
            np.linspace(200, 600 if load_location == LoadLocation.EDGE else 800, 100),
        )
    )

    f_space, p_space = np.meshgrid(f_space, p_space)
    f_space = f_space.ravel()
    p_space = p_space.ravel()
    t_space = t_12(f_12=f_space, p=p_space, load_location=load_location)

    mask = np.logical_not(np.isnan(t_space))

    f_space = f_space[mask]
    p_space = p_space[mask]
    t_space = t_space[mask]

    fig, ax = plt.subplots()
    # ax.tricontourf(f_space, t_space, p_space)
    ax.scatter(f_space, t_space, c=p_space, cmap="viridis", s=4)
    fig.colorbar(ax.collections[0], label="P (kN)")

    ax.set_xlabel("F_1" if load_location == LoadLocation.INTERNAL else "F_2")
    ax.set_ylabel("Thickness (mm)")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 600)
    ax.grid(visible=True)
    ax.axhspan(400, 600, color="grey", alpha=0.34)
    ax.text(
        x=3,
        y=500,
        s="Use computer analysis\nin shaded area.",
        bbox={"facecolor": "lightgrey", "edgecolor": "lightgrey"},
        fontsize=8,
    )

    for p in data["p"].unique():
        line_data = data.filter(pl.col("p") == p)
        ax.plot(line_data["f"], line_data["t"], color="black", label=f"p = {p} kN")

    ax.set_title(
        f"{'T_1' if load_location == LoadLocation.INTERNAL else 'T_2'} "
        + f"vs {'F_1' if load_location == LoadLocation.INTERNAL else 'F_2'}"
    )

    plt.show()


@lru_cache(maxsize=None)
def _t_3_data() -> pl.DataFrame:
    """
    Get the t_3 data into a DataFrame for easy use later.
    """

    data = {}

    data[LoadLocation.INTERNAL] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht13_t3_interior"
    )
    data[LoadLocation.EDGE] = pl.read_excel(
        _DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht13_t3_edge"
    )

    return data


def t_3(*, f_3: float, load_location: LoadLocation) -> float:
    """
    Calculate the required slab thickness for wheel loads.
    """

    data = _t_3_data()[load_location]

    if f_3 < data["f_3"].min():
        raise ValueError(
            f"f_3 value of {f_3} is less than "
            + f"the minimum value of {data['f_3'].min():.0f}"
        )

    if f_3 > data["f_3"].max():
        raise ValueError(
            f"f_3 value of {f_3} is greater than "
            + f"the maximum value of {data['f_3'].max():.0f}"
        )

    return np.interp(f_3, data["f_3"], data["t_3"])


def plot_t_3():
    """
    Plot the t_3 data.

    Primarily useful for debugging.
    """

    data = _t_3_data()

    fig, ax = plt.subplots()

    ax.set_prop_cycle(AGILITUS_COLORS)

    ax.plot(
        data[LoadLocation.INTERNAL]["f_3"],
        data[LoadLocation.INTERNAL]["t_3"],
        label="Interior",
    )
    ax.plot(
        data[LoadLocation.EDGE]["f_3"], data[LoadLocation.EDGE]["t_3"], label="Edge"
    )

    ax.set_xlabel("F_3")
    ax.set_ylabel("Thickness (mm)")
    ax.legend()
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 600)
    ax.grid(visible=True)

    ax.axhspan(400, 600, color="grey", alpha=0.34)
    ax.text(
        x=80,
        y=500,
        s="Use computer analysis\nin shaded area.",
        bbox={"facecolor": "lightgrey", "edgecolor": "lightgrey"},
        fontsize=8,
    )

    ax.set_title("Thickness vs F_3")

    plt.show()


@lru_cache(maxsize=None)
def _t_4_data() -> pl.DataFrame:
    """
    Get the t_4 data into a DataFrame for easy use later.

    Notes
    -----
    - The data is interpolated from CCAA T48 Chart 1.4.
    - A separate method is used so that the call into the spreadsheet can be cached.
    """

    return pl.read_excel(_DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht14_t4")


def t_4(*, f_4: float) -> float | np.ndarray:
    """
    Calculate the required slab thickness for distributed loads.

    Notes
    -----
    - The data is interpolated from CCAA T48 Chart 1.4.

    Parameters
    ----------
    f_4 : float
        The equivalent stress factor, F_4

    Returns
    -------
    float
        The required slab thickness, in mm
    """

    data = _t_4_data()

    if f_4 < data["f4"].min():  # type: ignore
        raise ValueError(
            f"f_4 value of {f_4} is less than "
            + f"the minimum value of {data['f4'].min():.0f}"
        )

    if f_4 > data["f4"].max():  # type: ignore
        raise ValueError(
            f"f_4 value of {f_4} is greater than "
            + f"the maximum value of {data['f4'].max():.0f}"
        )

    return np.interp(f_4, data["f4"], data["t4"])


def plot_t_4():
    """
    Plot the t_4 data.

    Primarily useful for debugging.
    """

    data = _t_4_data()

    fig, ax = plt.subplots()

    ax.set_prop_cycle(AGILITUS_COLORS)

    ax.plot(data["f4"], data["t4"], label="Thickness")

    ax.set_xlabel("F_4")
    ax.set_ylabel("Thickness (mm)")
    ax.legend()
    ax.set_xlim(30, 100)
    ax.set_ylim(0, 600)
    ax.grid(visible=True)
    ax.axhspan(400, 600, color="grey", alpha=0.34)
    ax.text(
        x=80,
        y=500,
        s="Use computer analysis\nin shaded area.",
        bbox={"facecolor": "lightgrey", "edgecolor": "lightgrey"},
        fontsize=8,
    )

    ax.set_title("Thickness vs F_4")

    plt.show()


def t_reqd(
    *,
    magnitude: float,
    x: float,
    load_type: LoadingType,
    load_location: LoadLocation,
    f_c: float,
    f_all: float,
    e_sx: float,
    h: float,
) -> float:
    f_e_calc = f_e(e_sx=e_sx, load_type=load_type, load_location=load_location)
    f_h_calc = f_h(h=h, load_type=load_type, load_location=load_location)
    f_s_calc = f_s(x=x, load_type=load_type, load_location=load_location)
    k_3_calc = k_3(load_location=load_location)
    k_4_calc = k_4(f_c=f_c)

    if load_type == LoadingType.WHEEL:
        f_12_calc = f_12(
            f_all=f_all,
            f_e12=f_e_calc,
            f_h12=f_h_calc,
            f_s12=f_s_calc,
            k_3=k_3_calc,
            k_4=k_4_calc,
        )

        return float(t_12(f_12=f_12_calc, p=magnitude, load_location=load_location))

    raise NotImplementedError()


@lru_cache(maxsize=None)
def _k_s_data():
    """
    Get the k_s data from the spreadsheet and into a dataframe for easy use later.

    Notes
    -----
    - The data is interpolated from CCAA T48 Chart C1.
    - A separate method is used so that the call into the spreadsheet can be cached.
    """

    return pl.read_excel(_DATA_PATH / Path("ccaa_t48_data.xlsx"), sheet_name="cht_c1")


def k_s_from_cbr(cbr: float) -> float:
    """
    Get the soil modulus of subgrade reaction from the CBR.

    Parameters
    ----------
    cbr : float
        The CBR value in %.

    Returns
    -------
    float
        The soil modulus of subgrade reaction, in kPa/mm.
    """

    data = _k_s_data()

    if cbr < data["cbr"].min():  # type: ignore
        raise ValueError(
            f"CBR value of {cbr} is less than "
            + f"the minimum value of {data['cbr'].min():.0f}"
        )

    if cbr > data["cbr"].max():  # type: ignore
        raise ValueError(
            f"CBR value of {cbr} is greater than "
            + f"the maximum value of {data['cbr'].max():.0f}"
        )

    return np.interp(cbr, data["cbr"], data["k_s"])


def plot_k_s_vs_cbr():
    """
    Plot the k_s vs cbr data.

    Primarily useful for debugging.
    """

    data = _k_s_data()

    fig, ax = plt.subplots()

    ax.plot(data["cbr"], data["k_s"])
    ax.set_xlabel("CBR (%)")
    ax.set_ylabel("$k_{s}$ (kPa/mm)")
    ax.set_title("$k_{s}$ vs CBR")
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 220)
    ax.set_xscale("log")
    ax.grid(visible=True)

    plt.show()


class Load:
    def __init__(
        self,
        *,
        load_type: LoadingType,
        magnitude: float,
        normalising_length: float,
        no_cycles: float,
    ):
        """
        Initialise a Load object.

        Parameters
        ----------
        load_type : LoadingType
            The type of load (e.g. wheel load, point load, distributed load).
        load_location : LoadLocation
            The location where the load is applied.
        magnitude : float
            The magnitude of the load:
            - For wheel loads or point loads: load in kN
            - For distributed loads: pressure in kPa
        normalising_length : float
            The normalising length:
            - For wheel loads or point loads: spacing between loads
            - For distributed loads: width of load or aisle
        no_cycles : float
            The number of load cycles.
        """

        self._load_type = load_type
        self._magnitude = magnitude
        self._normalising_length = normalising_length
        self._no_cycles = no_cycles

    @property
    def load_type(self) -> LoadingType:
        """
        The type of load.
        """

        return self._load_type

    @property
    def magnitude(self) -> float:
        """
        The load magnitude.

        Notes
        -----
        - For wheel loads or point loads this is in kN.
        - For distributed loads this is in kPa.
        """

        return self._magnitude

    @property
    def normalising_length(self) -> float:
        """
        The normalising length.

        Notes
        -----
        - For wheel loads or point loads this is the spacing.
        - For distributed loads this is the width of the load or the aisle.
        """

        return self._normalising_length

    @property
    def no_cycles(self) -> float:
        """
        The number of cycles or the load.
        """

        return self._no_cycles

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Load):
            return False

        return (
            self.load_type == other.load_type
            and self.magnitude == other.magnitude
            and self.normalising_length == other.normalising_length
            and self.no_cycles == other.no_cycles
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}: "
            + f"Load Type: {self.load_type}, "
            + f"Load Magnitude: {self.magnitude}"
            + f"{'kPa' if self.load_type == LoadingType.DISTRIBUTED else 'kN'}, "
            + f"Normalising Length: {self.normalising_length}, "
            + f"No. cycles: {self.no_cycles}."
        )


class Slab:
    """
    A class to handle calculating slab thicknesses

    Notes
    -----
    - Slab objects are immutable. Methods that modify internal state will return new
      Slab objects.
    """

    def __init__(self, *, f_c: float, f_tf: float, thickness: float):
        """
        Initialise a Slab.

        Parameters
        ----------
        f_c : float
            The characteristic strength of the concrete, in MPa.
        f_tf : float
            The flexural tensile strength of the concrete, in MPa.
        thickness : float
            The thickness of the slab. In m.
        """

        self._f_c = f_c
        self._f_tf = f_tf
        self._thickness = thickness

    def _copy(self) -> "Slab":
        """
        Create a copy of the slab object
        """

        return Slab(
            f_c=self.f_c,
            f_tf=self.f_tf,
            thickness=self.thickness,
        )

    @property
    def f_c(self) -> float:
        """
        The characteristic strength of hte concrete. in MPa.
        """

        return self._f_c

    @property
    def f_tf(self) -> float:
        """
        The flexural tensile strength of the concrete. In MPa.
        """

        return self._f_tf

    @property
    def thickness(self) -> float:
        """
        The thickness of the slab, in m.
        """

        return self._thickness

    def __repr__(self):
        return (
            f"{type(self).__name__}: "
            + f"f_tf: {self.f_tf:.2f}MPa, "
            + f"t: {self.thickness:.3f}m"
        )


class CCAA_T48:  # noqa: N801
    """
    A class to carry out design checks to CCAA T48
    """

    def __init__(
        self,
        *,
        slab: Slab,
        material_factor: MaterialFactor = MaterialFactor.CONSERVATIVE,
        loads: dict[str, Load] | None = None,
        soil_profile: SoilProfile | None = None,
    ):
        """
        Create a design check to CCAA T48.

        Parameters
        ----------

        """

        self._slab = slab
        self._material_factor = material_factor
        self._loads = loads if loads is not None else {}
        self._soil_profile = soil_profile

        self._k_s = None  # modulus of subgrade reaction.
        self._e_ss = None  # short term young's modulus.
        self._e_sl = None  # long term young's modulus

    @property
    def slab(self) -> Slab:
        """
        The soil profile below the slab.
        """

        return self._slab

    @property
    def material_factor(self) -> MaterialFactor:
        """
        The material factor. Used to select k_1 as per Table 1.16 of CCAA T48.
        """

        return self._material_factor

    @property
    def loads(self) -> dict[str, Load]:
        return self._loads

    @property
    def soil_profile(self) -> SoilProfile | None:
        return self._soil_profile

    def add_load(
        self,
        *,
        load_id: str,
        load_type: LoadingType,
        magnitude: float,
        normalising_length: float,
        no_cycles: float,
    ) -> "CCAA_T48":
        """
        Add a load to the CCAA_T48 object.

        Notes
        -----
        - CCAA_T48 objects are immutable, so this method returns a new CCAA_T48 object.

        Parameters
        ----------
        load_id : str
            The ID of the load.
        load_type : LoadingType
            The type of load.
        magnitude : float
            The load magnitude.
            Should be kN for wheel and point loads and kPa for distributed loads.
        normalising_length : float
            The normalising length.
        no_cycles: float
            The no. of cycles in the load.

        Returns
        -------
        CCAA_T48
            A new CCAA_T48 object with the load added.
        """

        new_check = self._copy()

        if load_id in new_check.loads:
            raise ValueError(f"Load {load_id} already exists.")

        load = Load(
            load_type=load_type,
            magnitude=magnitude,
            normalising_length=normalising_length,
            no_cycles=no_cycles,
        )
        new_check.loads[load_id] = load

        return new_check

    def add_loads(self, *, loads: dict[str, Load]) -> "CCAA_T48":
        """
        Add multiple loads to the CCAA_T48 object.

        Notes
        -----
        - CCAA_T48 objects are immutable, so this method returns a new CCAA_T48 object.

        Parameters
        ----------
        loads : dict[str, Load]
            The loads to add. Should be a dict of the form {load_id: Load}.

        Returns
        -------
        CCAA_T48
            A new CCAA_T48 object with the loads added.
        """

        loads = deepcopy(loads)

        for load_id in loads:
            if load_id in self.loads:
                raise ValueError(f"Load {load_id} already exists.")

        new_slab = self._copy()
        new_slab._loads = new_slab.loads | loads

        return new_slab

    def _copy(self) -> "CCAA_T48":
        return CCAA_T48(
            slab=deepcopy(self.slab),
            material_factor=deepcopy(self.material_factor),
            loads=deepcopy(self.loads),
            soil_profile=deepcopy(self.soil_profile),
        )

    @property
    def k_s(self) -> float | None:
        """
        The subgrade modulus in kPa / mm or MPa / m.

        Returns
        -------
        float | None
            The subgrade modulus in kPa/mm or MPa/m. Returns None if not set.
        """

        return self._k_s

    def set_k_s(
        self, *, k_s: float | None = None, cbr: float | None = None
    ) -> "CCAA_T48":
        """

        Parameters
        ----------
        k_s : float | None
            The subgrade modulus in kPa/mm or MPa/m. If provided,
            this value will be used directly. If not provided and cbr is provided,
            k_s will be calculated from the CBR value.
        cbr : float | None
            The California Bearing Ratio (CBR) value in percent. If provided and k_s is None,
            k_s will be calculated from this value using the relationship from CCAA T48.

        Returns
        -------
        CCAA_T48
            A new CCAA_T48 object with the k_s value set.

        Raises
        ------
        Raises an error if both k_s and cbr are None.
        """

        if k_s is None and cbr is None:
            raise ValueError("Must provide either k_s or cbr. Both were None.")

        check = self._copy()

        if k_s is None and cbr is not None:
            k_s = k_s_from_cbr(cbr=cbr)

        check._k_s = k_s
        return check

    def set_e_s(self, *, e_ss: float, e_sl: float) -> "Slab":
        """
        Set the Young's modulus values. Overrides the values calculated from the
        soil_profile class.

        Parameters
        ----------
        e_ss : float
        e_sl : float
        """

        check = self._copy()
        check._e_ss = e_ss
        check._e_sl = e_sl

        return check

    def e_ss(self, *, load_id: str):
        if self._e_ss is not None:
            return self._e_ss

        load = self.loads[load_id]

        return self.soil_profile.e_ss(
            normalising_length=load.normalising_length, loading_type=load.load_type
        )

    def e_sl(self, *, load_id: str):
        if self._e_sl is not None:
            return self._e_ss

        load = self.loads[load_id]

        return self.soil_profile.e_sl(
            normalising_length=load.normalising_length, loading_type=load.load_type
        )

    def k_1(self, *, load_id) -> float | None:
        """
        Calculate the material factor, k_1 as per Table 1.16 of CCAA T48.
        """

        if len(self.loads) == 0:
            return None

        load = self.loads[load_id]

        return k_1(loading_type=load.load_type, material_factor=self.material_factor)

    def k_2(self, *, load_id) -> float | None:
        """
        Calculate the load repetition factor, k_2 as per Table 1.17 of CCAA T48.
        """

        if len(self.loads) == 0:
            return None

        load = self.loads[load_id]

        return k_2(no_cycles=load.no_cycles, load_type=load.load_type)

    def f_all(self, *, load_id) -> float | None:
        """
        Calculate the allowable concrete stress f_all for a given load case.

        Parameters
        ----------
        load_id : str
            The ID of the load.

        Returns
        -------
        float
            The allowable concrete stress f_all. In MPa.
        """

        if self.loads is None:
            return None

        if load_id not in self.loads:
            raise ValueError(f"Load {load_id!r} is not in self.loads")

        k_1_l = self.k_1(load_id=load_id)
        k_2_l = self.k_2(load_id=load_id)

        if k_1_l is None or k_2_l is None:
            raise ValueError("Error calculating k_1 and k_2")

        return self.slab.f_tf * k_1_l * k_2_l

    def t_reqd(
        self, *, load_id: str, load_location: LoadLocation, load_duration: LoadDuration
    ) -> float:
        load = self.loads[load_id]
        e_sx = (
            self.e_ss(load_id=load_id)
            if load_duration == LoadDuration.SHORT
            else self.e_sl(load_id=load_id)
        )

        return t_reqd(
            magnitude=load.magnitude,
            x=load.normalising_length,
            load_type=load.load_type,
            load_location=load_location,
            f_c=self.slab.f_c,
            f_all=self.f_all(load_id=load_id),
            e_sx=e_sx,
            h=self.soil_profile.h_total,
        )
