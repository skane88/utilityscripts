"""
File to contain some basic AS1170.2 helper methods for working with wind loads
"""

from pathlib import Path

import numpy as np
import toml

from utilityscripts.multiinterp import multi_interp

FILE_PATH = Path(__file__)
DEFAULT_DATA_PATH = FILE_PATH.parent / Path("as1170_2.toml")
STANDARD_DATA = {}

MIN_TERRAIN_CATEGORY = 1.0
MAX_TERRAIN_CATEGORY = 4.0


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

        if (
            terrain_category < MIN_TERRAIN_CATEGORY
            or terrain_category > MAX_TERRAIN_CATEGORY
        ):
            raise ValueError(
                "Terrain Category expected to be between 1.0 and 4.0. "
                + f"Received {terrain_category}"
            )

        self.terrain_category = terrain_category

        self.shielding_data = shielding_data

    def v_r(self, *, r: float, ignore_f_x: bool = False):
        return v_r(wind_region=self.wind_region, r=r, ignore_m_c=ignore_f_x)

    def m_d(self, direction: float | str):
        return m_d(direction=direction, wind_region=self.wind_region)

    def m_z_cat(self, z):
        return m_zcat_basic(z=z, terrain_category=self.terrain_category)

    def m_s(self):
        if self.shielding_data is None:
            # if there is no shielding data then we can't calculate the shielding
            # parameter. The shielding multiplier can conservatively be taken to be 1.0
            return 1.0

        # TODO: Complete calculation of shielding.
        raise NotImplementedError()

    def m_t(self):
        # TODO: Add an M_t method
        return 1.0

    def m_lee(self):
        # TODO: Add an M_lee method
        return 1.0

    def v_sit(
        self, *, r: float, direction: float | str, z: float, ignore_f_x: bool = False
    ):
        v_r = self.v_r(r=r, ignore_f_x=ignore_f_x)
        m_d = self.m_d(direction=direction)
        m_zcat = self.m_z_cat(z=z)
        m_s = self.m_s()
        m_t = self.m_t()
        m_lee = self.m_lee()

        return v_r * m_d * m_zcat * m_s * m_t * m_lee


def v_r_no_f_x(*, a, b, r, k):
    """
    Calculate the basic windspeed for a wind region. Ignores parameters F_C or F_D,
    for those use method V_R

    :param a: Windspeed parameter 'a'
    :param b: Windspeed parameter 'b'
    :param r: The Average Recurrence Interval (ARI) of the windspeed.
    :param k: Windspeed parameter 'k'
    """

    return a - b * r**-k


def f_x(*, wind_region, r):
    """
    Calculate the cyclonic region factor F_C or F_D.

    :param wind_region: The wind region.
    :param r: The Average Recurrence Interval (ARI) of the windspeed.
    """

    f_x_min_r = STANDARD_DATA["2011"]["region_windspeed_parameters"][wind_region][
        "F_x_min_R"
    ]

    if r < f_x_min_r:
        return 1.0

    return STANDARD_DATA["2011"]["region_windspeed_parameters"][wind_region]["F_x"]


def m_c(*, wind_region, r, version: str = "2021"):
    """
    Calculate the climate change factor M_C.

    :param wind_region: The wind region.
    :param r: The Average Recurrence Interval (ARI) of the windspeed.
    :param version: The version of the standard to look up.
    """

    m_c_min_r = STANDARD_DATA[version]["region_windspeed_parameters"][wind_region][
        "M_c_min_R"
    ]

    if r < m_c_min_r:
        return 1.0

    return STANDARD_DATA[version]["region_windspeed_parameters"][wind_region]["M_c"]


def v_r(*, wind_region: str, r, version: str = "2021", ignore_m_c: bool = False):
    """
    Calculate the regional windspeed V_R based on the region and return period.

    :param wind_region: The wind region where the structure is located.
    :param r: The Average Recurrence Interval (ARI) of the windspeed.
    :param ignore_m_c: Ignore the cyclonic region factor M_c, or for older standards
        the uncertainty factor F_C or F_D?
    :return: The regional windspeed.
    """

    if len(STANDARD_DATA) == 0:
        init_standard_data()

    if ignore_m_c:
        f = 1.0
    else:
        f = (
            m_c(wind_region=wind_region, r=r)
            if version == "2021"
            else f_x(wind_region=wind_region, r=r)
        )

    a = STANDARD_DATA[version]["region_windspeed_parameters"][wind_region]["a"]
    b = STANDARD_DATA[version]["region_windspeed_parameters"][wind_region]["b"]
    k = STANDARD_DATA[version]["region_windspeed_parameters"][wind_region]["k"]
    v_min = STANDARD_DATA[version]["region_windspeed_parameters"][wind_region]["V_min"]

    return max(f * v_min, f * v_r_no_f_x(a=a, b=b, r=r, k=k))


def m_d(
    *, wind_region: str, direction: float | str, version: str = "2021"
) -> tuple[float, float]:
    """
    Return the wind direction multiplier for a given region and wind direction.

    :param wind_region: The wind region where the structure is located.
    :param direction: The direction as an angle between 0 and 360 degrees.
        0 / 360 = Wind from North
        90 = Wind from East
        180 = Wind from South
        270 = Wind from West
        Alternatively, use "ANY" to return the any direction value, or the cardinal
        values (e.g. "N", "NE", "E", ..., "NW").
    :return: The direction multiplier as a Tuple containing (M_d, M_d_cladding)
    """

    # first load some required data
    if len(STANDARD_DATA) == 0:
        init_standard_data()

    if isinstance(direction, str):
        direction = direction.upper()

    region_m_d_parameters = STANDARD_DATA[version]["region_direction_parameters"][
        wind_region
    ]
    wind_direction_defs = STANDARD_DATA["wind_direction_definitions"]

    f_not_clad = region_m_d_parameters["F_not_clad"]

    # next bail out early if the direction doesn't matter
    if direction == "ANY":
        m_d_clad = region_m_d_parameters[direction]
        m_d = m_d_clad * f_not_clad
        return m_d, m_d_clad

    # next if the user has provided a text direction (e.g. "NW") get the angle value.
    if direction in wind_direction_defs:
        direction = wind_direction_defs[direction][0]

    # now check that direction is within the range of 0-360
    direction = direction % 360

    # now build a numpy array to use numpy's interp functions.
    m_d_table = [
        [a, region_m_d_parameters[d]]
        for d, angles in wind_direction_defs.items()
        for a in angles
    ]

    m_d_table = np.array(m_d_table)
    # make sure to sort it correctly
    m_d_table = m_d_table[np.argsort(m_d_table[:, 0])]

    # now interpolate the value
    m_d_clad = np.interp(direction, m_d_table[:, 0], m_d_table[:, 1])
    m_d = m_d_clad * f_not_clad

    return m_d, m_d_clad


def m_zcat_basic(*, z, terrain_category) -> float:
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
    terrain_cats = np.array([float(k) for k in terrain_height_multipliers["data"]])

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
    m_zcat_all = np.array(list(terrain_height_multipliers["data"].values())).transpose()

    # interpolate for the case of a terrain category that is intermediate.
    # this returns a list of M_zcat at all input heights.
    m_zcat_for_terrain = multi_interp(
        x=terrain_category, xp=terrain_cats, fp=m_zcat_all
    ).flatten()

    # interpolate M_zcat for the specified height.
    return np.interp(z, xp=heights, fp=m_zcat_for_terrain)


def m_zcat_ave():
    """
    Calculate the terrain height factor M_zcat for the averaged case.

    :return:
    """

    # TODO: Implement M_zcat_ave
    raise NotImplementedError()


def shielding_parameter(
    *,
    h: float,
    building_data: list[tuple[float, float]] | None = None,
    h_s: float | None = None,
    b_s: float | None = None,
    n_s: float | None = None,
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


def m_s(shielding_parameter):
    """
    Determine the shielding multiplier.

    :param shielding_parameter: The shielding parameter s.
    :return: The shielding multiplier M_s
    """

    # first load some required data
    if len(STANDARD_DATA) == 0:
        init_standard_data()

    shielding_multiplier_data = STANDARD_DATA["shielding_multiplier"]

    xp = np.array([float(x) for x in shielding_multiplier_data])
    fp = np.array([float(y) for y in shielding_multiplier_data.values()])

    return np.interp(shielding_parameter, xp, fp)


def q_basic(v: float, *, rho_air: float = 1.2):
    """
    Calculate the basic wind pressure caused by a certain wind velocity.

    :param rho_air: The density of air. Must be in kg / m³.
    :param v: The wind velocity. Must be in m/s.
    :return: The basic wind pressure in Pa.
    """

    return 0.5 * rho_air * v**2


def pipe_wind_loads(cd, qz, d_max, d_ave, n_pipes):
    """
    Calculate wind loads on pipes in pipe racks.

    Based on Oil Search design criteria 100-SPE-K-0001.

    :param cd: Drag coefficient for largest pipe in the group.
    :param qz: The design wind pressure.
    :param d_max: The maximum pipe diameter.
    :param d_ave: The average pipe diameter.
    :param n_pipes: The number of pipes in the tray.
    """

    shielding = np.asarray(
        [0, 0.70, 1.19, 1.53, 1.77, 1.94, 2.06, 2.14, 2.20, 2.24, 2.27, 2.29, 2.30]
    )

    if n_pipes == 0:
        return 0.0

    n_pipes = max(n_pipes - 1, 12)

    shielding_factor = shielding[n_pipes]

    return qz * cd * (d_max + d_ave * shielding_factor)


def temp_windspeed(*, a, b, k, r_s, t):
    """
    Calculate a temporary design windspeed V_R, for structures with design lives of less than 1 year.

    :param a: Windspeed parameter a.
    :param b: Windspeed parameter b.
    :param k: Windspeed parameter k.
    :param r_s: The reference return period for a normal structure. Typ. taken from the Building Code.
    :param t: The number of reference periods in a year.
    """

    return a - b * (1 - (1 - (1 / r_s)) ** t) ** k
