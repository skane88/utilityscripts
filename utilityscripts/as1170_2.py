"""
File to contain some basic AS1170.2 helper methods for working with wind loads
"""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import toml
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


class WindSite:
    def __init__(self, wind_region: str, terrain_category: float, shielding_data=None):
        self.wind_region = wind_region

        if terrain_category < 1.0 or terrain_category > 4.0:
            raise ValueError(
                "Terrain Category expected to be between 1.0 and 4.0. "
                + f"Received {terrain_category}"
            )

        self.terrain_category = terrain_category

        self.shielding_data = shielding_data

    def V_R(self, R: float, ignore_F_x: bool = False):
        return V_R(wind_region=self.wind_region, R=R, ignore_F_x=ignore_F_x)

    def M_d(self, direction: Union[float, str]):
        return M_d(direction=float, wind_region=self.wind_region)

    def M_z_cat(self, z):
        return M_zcat_basic(z=z, terrain_category=self.terrain_category)

    def M_s(self):
        if self.shielding_data is None:
            # if there is no shielding data then we can't calculate the shielding
            # parameter. The shielding multiplier can conservatively be taken to be 1.0
            return 1.0

        # TODO: Complete calculation of shielding.
        raise NotImplementedError()

    def M_t(self):
        # TODO: Add an M_t method
        return 1.0

    def M_lee(self):
        # TODO: Add an M_lee method
        return 1.0

    def V_sit(
        self, R: float, direction: Union[float, str], z: float, ignore_F_x: bool = False
    ):
        V_R = self.V_R(R=R, ignore_F_x=ignore_F_x)
        M_d = self.M_d(direction=direction)
        M_zcat = self.M_z_cat(z=z)
        M_s = self.M_s()
        M_t = self.M_t()
        M_lee = self.M_lee()

        return V_R * M_d * M_zcat * M_s * M_t * M_lee


def V_R_no_F_x(*, a, b, R, k):
    """
    Calculate the basic windspeed for a wind region. Ignores parameters F_C or F_D,
    for those use method V_R

    :param a: Windspeed parameter 'a'
    :param b: Windspeed parameter 'b'
    :param R: The Average Recurrence Interval (ARI) of the windspeed.
    :param k: Windspeed parameter 'k'
    """

    return a - b * R**-k


def F_x(*, wind_region, R):
    """
    Calculate the cyclonic region factor F_C or F_D.

    :param wind_region: The wind region.
    :param R: The Average Recurrence Interval (ARI) of the windspeed.
    """

    F_x_min_R = STANDARD_DATA["region_windspeed_parameters"][wind_region]["F_x_min_R"]

    if R < F_x_min_R:
        return 1.0

    return STANDARD_DATA["region_windspeed_parameters"][wind_region]["F_x"]


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

    F = F_x(wind_region=wind_region, R=R) if not ignore_F_x else 1.0
    a = STANDARD_DATA["region_windspeed_parameters"][wind_region]["a"]
    b = STANDARD_DATA["region_windspeed_parameters"][wind_region]["b"]
    k = STANDARD_DATA["region_windspeed_parameters"][wind_region]["k"]
    V_min = STANDARD_DATA["region_windspeed_parameters"][wind_region]["V_min"]

    return max(V_min, F * V_R_no_F_x(a=a, b=b, R=R, k=k))


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
    :param terrain_category: The terrain category, as a number between 1 and 4.
        Floats are acceptable for intermediate categories.
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


def M_zcat_ave():
    """
    Calculate the terrain height factor M_zcat for the averaged case.

    :return:
    """

    # TODO: Implement M_zcat_ave
    raise NotImplementedError()


def shielding_parameter(
    *,
    h: float,
    building_data: List[Tuple[float, float]] = None,
    h_s: float = None,
    b_s: float = None,
    n_s: float = None,
):
    """
    Calculate the shielding parameter.

    :param h: The average roof height of the design building being considered.
    :param building_data: A list of tuples containing the height and width of buildings
        in the sector that are shielding the design building.
        If None, h_s, b_s and n_s must be explicitly specified.
    :param h_s: The average height of shielding buildings.
    :param b_s: The average width of shielding buildings.
    :param n_s: The no. of shielding buildings.
    :return: The shielding parameter s.
    """

    if building_data is not None:
        n_s = len(building_data)

        heights = [x[0] for x in building_data]
        widths = [x[1] for x in building_data]

        h_s = sum(heights) / n_s
        b_s = sum(widths) / n_s

    l_s = h * ((10 / n_s) + 5)

    return l_s / ((h_s * b_s) ** 0.5)


def M_s(shielding_parameter):
    """
    Determine the shielding multiplier.

    :param shielding_parameter: The shielding parameter s.
    :return: The shielding multiplier M_s
    """

    # first load some required data
    if len(STANDARD_DATA) == 0:
        init_standard_data()

    shielding_multiplier_data = STANDARD_DATA["shielding_multiplier"]

    xp = np.array([float(x) for x in shielding_multiplier_data.keys()])
    fp = np.array([float(y) for y in shielding_multiplier_data.values()])

    return np.interp(shielding_parameter, xp, fp)


def q_basic(*, V: float, rho_air: float = 1.2):
    """
    Calculate the basic wind pressure caused by a certain wind velocity.

    :param rho_air: The density of air. Must be in kg / m³.
    :param V: The wind velocity. Must be in m/s.
    :return: The basic wind pressure in Pa.
    """

    return 0.5 * rho_air * V**2
