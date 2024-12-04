"""
File to contain some basic AS1170.2 helper methods for working with wind loads
"""

import copy
from enum import StrEnum
from math import log10, radians, tan
from pathlib import Path

import numpy as np
import polars as pl

from utilityscripts.multiinterp import multi_interp

FILE_PATH = Path(__file__)
DEFAULT_DATA_PATH = FILE_PATH.parent / Path("as1170_2.toml")
DEFAULT_DATA_PATH_XLSX = FILE_PATH.parent / Path("as1170_2.xlsx")
STANDARD_DATA = {}

MIN_TERRAIN_CATEGORY = 1.0
MAX_TERRAIN_CATEGORY = 4.0


class WallType(StrEnum):
    IMPERMEABLE = "impermeable"
    PERMEABLE = "permeable"
    OPEN = "open"


def init_standard_data(*, file_path=None, overwrite: bool = False):
    global STANDARD_DATA

    if not overwrite and len(STANDARD_DATA) > 0:
        return

    if file_path is None:
        file_path = DEFAULT_DATA_PATH_XLSX

        sheet_set = {
            "wind_direction_definitions",
            "shielding_multiplier",
            "region_windspeed_parameters",
            "region_direction_parameters",
            "terrain_height_multipliers",
            "cpi_t5a",
            "cpi_t5b",
        }

        STANDARD_DATA = {
            sheet: pl.read_excel(source=file_path, sheet_name=sheet)
            for sheet in sheet_set
        }


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

    def v_r(
        self, *, return_period: float, ignore_f_x: bool = False, version: str = "2021"
    ):
        return v_r(
            wind_region=self.wind_region,
            return_period=return_period,
            ignore_m_c=ignore_f_x,
            version=version,
        )

    def m_d(self, direction: float | str, version: str = "2021"):
        return m_d_exact(
            direction=direction, wind_region=self.wind_region, version=version
        )

    def m_z_cat(self, z, version: str = "2021"):
        return m_zcat_basic(
            z=z, terrain_category=self.terrain_category, version=version
        )

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
        self,
        *,
        return_period: float,
        direction: float | str,
        z: float | None,
        ignore_f_x: bool = False,
        version: str = "2021",
    ) -> tuple[float, float]:
        """
        Calculate the site design velocity v_sit_beta for a WindSite.

        Parameters
        ----------
        return_period : float
            The design return period.
        direction : float | string
            The angle the wind is blowing from.
        z : float | None
            The height the wind is blowing from.
            If None, uses 10m
        ignore_f_x : bool
            Ignore the climate change multiplier.
        version : str
            The version of the standard to design for.

        Returns
        -------

        """

        if z is None:
            z = 10.0

        v_r = self.v_r(
            return_period=return_period, ignore_f_x=ignore_f_x, version=version
        )
        m_d = self.m_d(direction=direction, version=version)
        m_zcat = self.m_z_cat(z=z, version=version)
        m_s = self.m_s()
        m_t = self.m_t()
        m_lee = self.m_lee()

        base_v = v_r * m_zcat * m_s * m_t * m_lee

        return base_v * m_d[0], base_v * m_d[1]

    def __repr__(self):
        return (
            f"{type(self).__name__}(wind_region={self.wind_region!r}, "
            f"terrain_category={self.terrain_category}, "
            f"shielding_data={self.shielding_data!r})"
        )


class SimpleBuilding:
    def __init__(
        self,
        *,
        wind_site: WindSite,
        principal_axis: float,
        z_ave: float,
        x_max: float,
        y_max: float,
        roof_pitch: float,
        openings: list[tuple[int, float]],
        version: str = "2021",
    ):
        """

                  face 0
                ---------
                |   |   |
                |   |   |
        face 3  |   |   |  face 1
                |   |   |
                |   |   |
                |   |   |
                ---------
                 face 2

        NOTE: Roof is face 4.

        Parameters
        ----------
        wind_site: WindSite
            A windsite object to contain information about the site.
        principal_axis: float
            The direction of the building principal axis.
            Typically the direction of the roof ridge line.
        z_ave: float
            The average height of the roof line.
        x_max: float
            The length of the building in the principal direction.
        y_max: float
            The width of the building.
        roof_pitch: float
            The roof pitch.
        openings: list[tuple[int, float]]
            Any openings in the building.
            A list of tuples of the form: [(face, opening_area)]
            In region B2, C & D these will be assumed
            to be open when necessary to achieve conservative design.
        version : str
            The standard to check the building against.
        """

        self._wind_site = wind_site
        self._principal_axis = principal_axis % 360  # ensure not larger than 360.
        self._z_ave = z_ave
        self._x_max = x_max
        self._y_max = y_max
        self._roof_pitch = roof_pitch
        self._openings = openings

    @property
    def wind_site(self) -> WindSite:
        return self._wind_site

    @property
    def principal_axis(self) -> float:
        return self._principal_axis

    @property
    def z_ave(self) -> float:
        return self._z_ave

    @property
    def x_max(self) -> float:
        return self._x_max

    @property
    def y_max(self) -> float:
        return self._y_max

    @property
    def roof_pitch(self) -> float:
        return self._roof_pitch

    @property
    def half_span(self) -> float:
        return self.y_max / 2.0

    @property
    def roof_rise(self) -> float:
        """
        The rise of the roof.

        Returns
        -------
        float
        """

        return self.half_span * tan(radians(self.roof_pitch))

    @property
    def z_max(self) -> float:
        """
        The maximum height of the roof.

        Returns
        -------
        float
        """

        return self.z_ave + self.roof_rise / 2.0

    @property
    def z_eaves(self) -> float:
        """
        The height of the eaves.

        Returns
        -------
        float
        """

        return self.z_ave - self.roof_rise / 2.0

    @property
    def openings(self) -> list[tuple[int, float]]:
        return self._openings

    def add_opening(self, opening: tuple[int, float]) -> "SimpleBuilding":
        new_building = copy.deepcopy(self)
        new_building._openings.append(opening)

        return new_building

    @property
    def total_open(self) -> float:
        """
        The total open area on the building.

        Returns
        -------
        float
        """

        return sum([area for _, area in self.openings])

    def open_area_on_face(self, face: int):
        """
        Total open area on a face.

                  face 0
                ---------
                |   |   |
                |   |   |
        face 3  |   |   |  face 1
                |   |   |
                |   |   |
                |   |   |
                ---------
                 face 2

        NOTE: Roof is face 4.

        Parameters
        ----------
        face : int
            The face to get open area on.
        """

        return sum([area for f, area in self.openings if f == face])

    @property
    def design_angles(self):
        """
        Return the angles to design for each face.

        Returns
        -------
        A list of the form [a0, a1, a2, a3]
        """

        a1 = self.principal_axis
        a2 = (self.principal_axis + 90) % 360
        a3 = (a2 + 90) % 360
        a4 = (a3 + 90) % 360

        return [a1, a2, a3, a4]

    def face_angle(self, face: int):
        """
        The face design angle for a particular face.

        Parameters
        ----------
        face : int
            The face to get the angle of.

        Returns
        -------
        float
        """

        return self.design_angles[face]

    def m_d(self, version: str = "2021") -> list[tuple[float, float]]:
        """
        Calculate the value of M_d for each face.

        Returns
        -------
        A list of the form [M_d0, M_d1, M_d2, M_d3]
        """

        return [
            m_d_des(
                wind_region=self.wind_site.wind_region, direction=a, version=version
            )
            for a in self.design_angles
        ]

    def m_d_on_face(self, face: int, version: str = "2021") -> [float, float]:
        """
        The value of M_d on a given face.

        Parameters
        ----------
        face: int
            The face to check.
        version : str
            The version of the standard to check.

        Returns
        -------
        [float, float]
        """
        return m_d_des(
            wind_region=self.wind_site.wind_region,
            direction=self.face_angle(face=face),
            version=version,
        )

    def m_c_or_f_x(self, wind_region: str, return_period: float, version: str = "2021"):
        """
        Get the value of M_c or F_C / F_D as appropriate.

        Parameters
        ----------
        wind_region : str
            The wind region.
        return_period : float
            The return period.
        version : str
            The version of the standard to check.

        Returns
        -------
        float
        """
        return m_c_or_f_x(
            wind_region=wind_region, return_period=return_period, version=version
        )

    def v_sit_beta(
        self, return_period: float, z: float | None = None, version="2021"
    ) -> list[tuple[float, float]]:
        """
        Return the design windspeeds for each face of the building.

        Parameters
        ----------
        return_period : float
            The return period to determine the wind load for.
        z : float | None
            The height at which to get the design pressure.
            If None, determines the pressure at 10m.
        version : str
            The version of the standard to check.

        Returns
        -------
        list[tuple[float, float]]
        The design windspeeds as [(V_sit_beta_struct_0, V_sit_beta_clad_0),
        ... , (V_sit_beta_struct_3, V_sit_beta_clad_3)]
        """

        return [
            self.wind_site.v_sit(
                return_period=return_period, direction=d, z=z, version=version
            )
            for d in self.design_angles
        ]

    def v_sit_beta_face(
        self, face: int, return_period: float, z: float | None = None, version="2021"
    ) -> tuple[float, float]:
        """
        Calculate the design windspeed for a given face.

        Parameters
        ----------
        face : int
            The face to get the wind speed for.
        return_period : float
            The return period to determine the wind load for.
        z : float | None
            The height to calculate the design windspeed at.
            If None, uses 10m instead.
        version : str
            The version of the standard to check.

        Returns
        -------
        tuple[float, float]
        The design windspeeds on the face as a tuple (V_sit_beta_struct, V_sit_beta_clad)
        """

        return self.v_sit_beta(return_period=return_period, version=version)[face]


def v_r_no_f_x(*, a, b, return_period, k):
    """
    Calculate the basic windspeed for a wind region. Ignores parameters F_C or F_D,
    for those use method V_R

    :param a: Windspeed parameter 'a'
    :param b: Windspeed parameter 'b'
    :param return_period: The Average Recurrence Interval (ARI) of the windspeed.
    :param k: Windspeed parameter 'k'
    """

    return a - b * return_period**-k


def m_c_or_f_x(*, wind_region, return_period, version: str = "2021"):
    """
    Calculate the climate change factor M_C
    (2021 edition of standard) or F_C / F_D (2011 edition).

    :param wind_region: The wind region.
    :param return_period: The Average Recurrence Interval (ARI) of the windspeed.
    :param version: The version of the standard to look up.
    """

    filtered_df = STANDARD_DATA["region_windspeed_parameters"].filter(
        (pl.col("standard") == int(version)) & (pl.col("region") == wind_region)
    )

    m_c, m_c_min_r = filtered_df.select(["m_c", "m_c_min_r"]).row(0)

    if return_period < m_c_min_r:
        return 1.0

    return m_c


def v_r(
    *, wind_region: str, return_period, version: str = "2021", ignore_m_c: bool = False
):
    """
    Calculate the regional windspeed V_R based on the region and return period.

    :param wind_region: The wind region where the structure is located.
    :param r: The Average Recurrence Interval (ARI) of the windspeed.
    :param ignore_m_c: Ignore the cyclonic region factor M_c, or for older standards
        the uncertainty factor F_C or F_D?
    :return: The regional windspeed.
    """

    init_standard_data()

    f = (
        1.0
        if ignore_m_c
        else m_c_or_f_x(
            wind_region=wind_region, return_period=return_period, version=version
        )
    )

    # Filter the DataFrame based on 'standard' and 'region' columns
    filtered_df = STANDARD_DATA["region_windspeed_parameters"].filter(
        (pl.col("standard") == int(version)) & (pl.col("region") == wind_region)
    )

    # Extracting the required values from the filtered DataFrame
    a, b, k, v_min = filtered_df.select(["a", "b", "k", "v_min"]).row(0)

    return max(f * v_min, f * v_r_no_f_x(a=a, b=b, return_period=return_period, k=k))


def m_d_exact(
    *, wind_region: str, direction: float | str, version: str = "2021"
) -> tuple[float, float]:
    """
    Return the wind direction multiplier for a given region and wind direction.

    Does ot consider m_d in +/-45deg as required by AS1170.2 S2.3 - this should
    be considered by the user of this method.

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
    init_standard_data()

    if isinstance(direction, str):
        direction = direction.upper()

    wind_direction_defs = STANDARD_DATA["wind_direction_definitions"]

    filtered_df = STANDARD_DATA["region_direction_parameters"].filter(
        (pl.col("wind_region") == wind_region) & (pl.col("standard") == int(version))
    )

    f_not_clad = filtered_df.filter(pl.col("direction") == "F_not_clad")["m_d"][0]

    # next bail out early if the direction doesn't matter
    if direction == "ANY":
        m_d_clad = filtered_df.filter(pl.col("direction") == direction)["m_d"][0]
        m_d = m_d_clad * f_not_clad
        return m_d, m_d_clad

    # now throw away unneeded rows from the dataframe.
    filtered_df = filtered_df.filter(
        (pl.col("direction") != "ANY") & (pl.col("direction") != "F_not_clad")
    )

    # next if the user has provided a text direction (e.g. "NW") get the angle value.
    if direction in wind_direction_defs["direction"]:
        direction = wind_direction_defs.filter(pl.col("direction") == direction)[
            "angle"
        ][0]

    # now check that direction is within the range of 0-360
    direction = direction % 360

    # now build a numpy array to use numpy's interp functions.
    m_d_table = [
        [a, filtered_df.filter(pl.col("direction") == d)["m_d"][0]]
        for d, a in wind_direction_defs.iter_rows()
    ]
    m_d_table.append([360.0, m_d_table[0][1]])

    m_d_table = np.array(m_d_table)
    # make sure to sort it correctly
    m_d_table = m_d_table[np.argsort(m_d_table[:, 0])]

    # now interpolate the value
    m_d_clad = np.interp(direction, m_d_table[:, 0], m_d_table[:, 1])
    m_d = m_d_clad * f_not_clad

    return m_d, m_d_clad


def m_d_des(
    *,
    wind_region: str,
    direction: float | str,
    version: str = "2021",
    tolerance: float = 45,
) -> tuple[float, float]:
    """
    Determine the design value of the direction factor M_d,
    within +/- tolderance of direction.

    Parameters
    ----------
    wind_region : str
        The wind region.
    direction : float
        The wind direction
    version : str
        The standard to design for.
    tolerance : float
        The tolerance in the wind angle.
        Typically 45degrees.

    Returns
    -------
    tuple[float, float]
    Consisting of (M_d_struct, M_d_cladding)
    """

    if isinstance(direction, str):
        angles = [direction, direction, direction]
    else:
        angles = [
            (direction - tolerance) % 360,
            direction % 360,
            (direction + tolerance) % 360,
        ]

    m_d_vals = [
        m_d_exact(wind_region=wind_region, direction=a, version=version) for a in angles
    ]

    m_d_struct = max([m[0] for m in m_d_vals])
    m_d_clad = max([m[1] for m in m_d_vals])

    return (m_d_struct, m_d_clad)


def m_zcat_basic(*, z, terrain_category, version="2021") -> float:
    """
    Determine the basic terrain category M_zcat at a given height and terrain.

    Does not do any averaging for changing terrain in the windward direction.

    :param z: The height at which the windspeed is being assessed.
    :param terrain_category: The terrain category, as a number between 1 and 4.
        Floats are acceptable for intermediate categories.
    """

    # first load some required data
    init_standard_data()
    terrain_height_multipliers = STANDARD_DATA["terrain_height_multipliers"]

    # get the basic data into the function as np arrays as we will be interpolating
    terrain_cats = terrain_height_multipliers.filter(
        pl.col("standard") == int(version)
    )["terrain_cat"].unique()

    min_cat = min(terrain_cats)
    max_cat = max(terrain_cats)

    if terrain_category < min_cat or terrain_category > max_cat:
        raise ValueError(
            f"Terrain Category {terrain_category} is outside the range "
            + f"{min_cat} to {max_cat}"
        )

    # load the M_zcat data for all terrain types
    m_z_cat_data = terrain_height_multipliers.filter(  # noqa: PD010
        pl.col("standard") == int(version)
    ).pivot("terrain_cat", index="height", values="m_z_cat")

    heights = np.array(m_z_cat_data["height"].unique())

    max_height = max(heights)
    min_height = min(heights)

    if z < min_height or z > max_height:
        raise ValueError(
            f"Height {z} is outside the range " + f"{min_height} to {max_height}"
        )

    m_zcat_all = np.array(m_z_cat_data.drop("height"))

    # interpolate for the case of a terrain category that is intermediate.
    # this returns a list of M_zcat at all input heights.
    m_zcat_for_terrain = multi_interp(
        x=terrain_category, xp=terrain_cats, fp=m_zcat_all
    ).flatten()

    # interpolate M_zcat for the specified height.
    return float(np.interp(z, xp=heights, fp=m_zcat_for_terrain))


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
    building_data: list[tuple[float, float]],
):
    """
    Calculate the shielding parameter.

    :param h: The average roof height of the design building being considered.
    :param building_data: A list of tuples containing the height and width of buildings
        in the sector that are shielding the design building.
        If None, h_s, b_s and n_s must be explicitly specified.
    :param h_s: The average height of shielding buildings.
    :return: The shielding parameter s.
    """

    n_s = len([b for b in building_data if b[0] >= h])

    h_s = sum([b[0] for b in building_data if b[0] >= h]) / n_s
    b_s = sum([b[1] for b in building_data if b[0] >= h]) / n_s

    l_s = h * ((10 / n_s) + 5)

    return l_s / ((h_s * b_s) ** 0.5)


def m_s(shielding_parameter):
    """
    Determine the shielding multiplier.

    :param shielding_parameter: The shielding parameter s.
    :return: The shielding multiplier M_s
    """

    # first load some required data
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


def k_v(*, open_area, volume):
    """
    Calculate the volume coefficient K_v as per AS1170.2

    :param open_area: The open area on the critical face.
    :param volume: The volume of the enclosed space.
    """

    alpha = 100 * (open_area ** (3 / 2)) / volume

    if alpha < 0.09:  # noqa: PLR2004
        return 0.85

    if alpha > 3:  # noqa: PLR2004
        return 1.085

    return 1.01 + 0.15 * log10(alpha)
