"""
File to contain some basic AS1170.2 helper methods for working with wind loads
"""

from pathlib import Path
from typing import Tuple, Union

import toml
import numpy as np

from multiinterp import multi_interp

FILE_PATH = Path(__file__)
DEFAULT_DATA_PATH = FILE_PATH.parent / Path("as1170_2.toml")
STANDARD_DATA = {}


def init_standard_data(*, file_path=None):
    """
    Initialise the standard_data dictionary if required.
    :param file_path: An optional filepath rather than using DEFAULT_DATA
    """

    global STANDARD_DATA

    if file_path is None:
        file_path = DEFAULT_DATA_PATH

    STANDARD_DATA = toml.load(f=file_path)


def V_R_basic(*, a, b, R, k):
    """
    Calculate the basic windspeed for a wind region. Ignores parameters F_C or F_D, for those use method V_R

    :param a: Windspeed parameter 'a'
    :param b: Windspeed parameter 'b'
    :param R: The Average Recurrence Interval (ARI) of the windspeed.
    :param k: Windspeed parameter 'k'
    """

    return a - b * R ** -k


def F_x(*, wind_region, R):
    """
    Calculate the cyclonic region factor F_C or F_D.

    :param wind_region: The wind region.
    :param R: The Average Recurrence Interval (ARI) of the windspeed.
    """

    F_x_val = STANDARD_DATA["region_windspeed_parameters"][wind_region]["F_x"]
    F_x_min_R = STANDARD_DATA["region_windspeed_parameters"][wind_region]["F_x_min_R"]

    if R < F_x_min_R:
        return 1.0

    return F_x_val


def V_R(*, wind_region: str, R, ignore_F_x: bool = False):
    """
    Calculate the regional windspeed V_R based on the region and return period.

    :param wind_region: The wind region where the structure is located.
    :param R: The Average Recurrence Interval (ARI) of the windspeed.
    :param ignore_F_x: Ignore the cyclonic region factor F_C or F_D?
    :return: The regional windspeed.
    """

    if len(STANDARD_DATA) == 0:
        init_standard_data()

    if not ignore_F_x:
        F = F_x(wind_region=wind_region, R=R)
    else:
        F = 1.0

    a = STANDARD_DATA["region_windspeed_parameters"][wind_region]["a"]
    b = STANDARD_DATA["region_windspeed_parameters"][wind_region]["b"]
    k = STANDARD_DATA["region_windspeed_parameters"][wind_region]["k"]
    V_min = STANDARD_DATA["region_windspeed_parameters"][wind_region]["V_min"]

    return max(V_min, F * V_R_basic(a=a, b=b, R=R, k=k))


def M_d(*, wind_region: str, direction: Union[float, str]) -> Tuple[float, float]:
    """
    Return the wind direction multiplier for a given region and wind direction.

    :param wind_region: The wind region where the structure is located.
    :param direction: The direction as an angle between 0 and 360 degrees.
        0 / 360 = Wind from North
        90 = Wind from East
        180 = Wind from South
        270 = Wind from West
        Alternatively, use "any" to return the any direction value.
    :return: The direction multiplier as a Tuple containing (M_d, M_d_cladding)
    """

    # first load some required data
    if len(STANDARD_DATA) == 0:
        init_standard_data()

    if isinstance(direction, str):
        direction = direction.lower()

    region_M_d_parameters = STANDARD_DATA["region_direction_parameters"][wind_region]
    wind_direction_defs = STANDARD_DATA["wind_direction_definitions"]

    F_not_clad = region_M_d_parameters["F_not_clad"]

    # next bail out early if the direction doesn't matter
    if direction == "any":
        M_d_clad = region_M_d_parameters[direction]
        M_d = M_d_clad * F_not_clad
        return (M_d, M_d_clad)

    # now check that direction is within the range of 0-360
    direction = direction % 360

    # now build a numpy array to use numpy's interp functions.
    M_d_table = []
    for d, angles in wind_direction_defs.items():
        for a in angles:
            M_d_table.append([a, region_M_d_parameters[d]])

    M_d_table = np.array(M_d_table)
    # make sure to sort it correctly
    M_d_table = M_d_table[np.argsort(M_d_table[:, 0])]

    # now interpolate the value
    M_d_clad = np.interp(direction, M_d_table[:, 0], M_d_table[:, 1])
    M_d = M_d_clad * F_not_clad

    return (M_d, M_d_clad)


def M_zcat_basic(*, z, terrain_category) -> float:
    """
    Determine the basic terrain category M_zcat at a given height and terrain.

    Does not do any averaging for changing terrain in the windward direction.

    :param z: The height at which the windspeed is being assessed.
    :param terrain_category: The terrain category, as a number between 1 and 4. Floats are
        acceptable for intermediate categories.
    """

    # first load some required data
    if len(STANDARD_DATA) == 0:
        init_standard_data()
    terrain_height_multipliers = STANDARD_DATA["terrain_height_multipliers"]

    # get the basic data into the function as np arrays as we will be interpolating
    heights = np.array(terrain_height_multipliers["heights"])
    terrain_cats = np.array(
        [float(k) for k in terrain_height_multipliers["data"].keys()]
    )

    min_cat = min(terrain_cats)
    max_cat = max(terrain_cats)

    if terrain_category < min_cat or terrain_category > max_cat:
        raise ValueError(
            f"Terrain Category {terrain_category} is outside the range "
            + f"{min_cat} to {max_cat}"
        )

    max_height = max(heights)
    min_height = min(heights)

    if z < min_height or z > max_height:
        raise ValueError(
            f"Height {z} is outside the range " + f"{min_height} to {max_height}"
        )

    # load the M_zcat data for all terrain types
    M_zcat_all = np.array(
        [t for t in terrain_height_multipliers["data"].values()]
    ).transpose()

    # interpolate for the case of a terrain category that is intermediate.
    # this returns a list of M_zcat at all input heights.
    M_zcat_for_terrain = multi_interp(
        x=terrain_category, xp=terrain_cats, fp=M_zcat_all
    ).flatten()

    # interpolate M_zcat for the specified height.
    return np.interp(z, xp=heights, fp=M_zcat_for_terrain)
