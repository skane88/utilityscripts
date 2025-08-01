"""
File to contain some basic AS1170.2 helper methods for working with wind loads
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import cos, log10, radians
from numbers import Real
from pathlib import Path

import numpy as np
import polars as pl
from scipy.interpolate import RegularGridInterpolator

# TODO: remove multi_interp and use scipy interpolator instead
from utilityscripts.multiinterp import multi_interp

FILE_PATH = Path(__file__)
DEFAULT_DATA_PATH = FILE_PATH.parent / Path("as1170_2.toml")
DEFAULT_DATA_PATH_XLSX = FILE_PATH.parent / Path("as1170_2_data.xlsx")
STANDARD_DATA = {}

MIN_TERRAIN_CATEGORY = 1.0
MAX_TERRAIN_CATEGORY = 4.0


class CardinalDirection(StrEnum):
    """
    The valid cardinal directions
    """

    N = "N"
    NE = "NE"
    E = "E"
    SE = "SE"
    S = "S"
    SW = "SW"
    W = "W"
    NW = "NW"
    ANY = "ANY"


class StandardVersion(StrEnum):
    AS1170_2_2011 = "AS/NZS1170.2-2011"
    AS1170_2_2021 = "AS/NZS1170.2-2021"


class WallType(StrEnum):
    IMPERMEABLE = "impermeable"
    PERMEABLE = "permeable"
    OPEN = "open"


class FaceType(StrEnum):
    WINDWARD = "windward"
    LEEWARD = "leeward"
    SIDE = "side"
    ROOF = "roof"


class RoofType(StrEnum):
    HIP = "hip"
    GABLE = "gable"


class SectionType(StrEnum):
    CIRCULAR = "circular"
    SHARP = "sharp"


class WindRegion(StrEnum):
    """
    Valid wind regions as per AS1170.2

    Notes
    -----
    Some of these regions are not valid in every version of the standard.
    """

    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"
    A5 = "A5"
    A6 = "A6"
    A7 = "A7"
    B = "B"
    B1 = "B1"
    B2 = "B2"
    NZ1 = "NZ1"
    NZ2 = "NZ2"
    NZ3 = "NZ3"
    NZ4 = "NZ4"
    C = "C"
    D = "D"
    W = "W"


@dataclass(slots=True, kw_only=True)
class M_d:  # noqa: N801
    """
    Stores the direction factor for structure and cladding together
    """

    struct: float
    cladding: float


@dataclass(slots=True, kw_only=True)
class C_pe:  # noqa: N801
    """
    Stores pressure coefficients (max, min)
    """

    c_pe_min: float
    c_pe_max: float


def valid_region(region: WindRegion, version: StandardVersion) -> bool:
    """
    Check if a wind region is valid for a given version of the standard.

    Parameters
    ----------
    region : WindRegion
        The wind region to check.
    version : StandardVersion
        The version of the standard to check.

    Returns
    -------
    bool
        True if the wind region is valid for the given version of the standard,
        False otherwise.
    """

    valid_regions = {
        StandardVersion.AS1170_2_2011: [
            WindRegion.A1,
            WindRegion.A2,
            WindRegion.A3,
            WindRegion.A4,
            WindRegion.A5,
            WindRegion.A6,
            WindRegion.A7,
            WindRegion.B,
            WindRegion.C,
            WindRegion.D,
            WindRegion.W,
        ],
        StandardVersion.AS1170_2_2021: [
            WindRegion.A0,
            WindRegion.A1,
            WindRegion.A2,
            WindRegion.A3,
            WindRegion.A4,
            WindRegion.A5,
            WindRegion.B1,
            WindRegion.B2,
            WindRegion.C,
            WindRegion.D,
            WindRegion.NZ1,
            WindRegion.NZ2,
            WindRegion.NZ3,
            WindRegion.NZ4,
        ],
    }

    return region in valid_regions[version]


def init_standard_data(*, file_path: Path | None = None, overwrite: bool = False):
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
            "cpe_t5_2b",
            "cpe_t5_2c",
            "cpe_t5_3",
            "k_a",
            "app_c_k_ar",
            "app_c_fig_c2",
        }

        STANDARD_DATA = {
            sheet: pl.read_excel(source=file_path, sheet_name=sheet)
            for sheet in sheet_set
        }


class WindSite:
    """
    A class to represent a site for wind design.
    """

    def __init__(
        self,
        wind_region: WindRegion,
        terrain_category: float,
        shielding_data: list[tuple[float, float]] | None = None,
    ):
        """
        Initialise a WindSite object.

        Notes
        -----
        AS1170.2 only defines terrain categories 1.0, 2.0, 2.5, 3.0 and 4.0.
        Intermediate terrain categories will be linearly interpolated.

        Shielding is not yet implemented.

        Parameters
        ----------
        wind_region : str
            The wind region as per AS1170.2.
        terrain_category : float
            The terrain category. Should be between 1.0 and 4.0.
        shielding_data : list[tuple[float, float]]
            The shielding data.
        """

        self._wind_region = wind_region

        if (
            terrain_category < MIN_TERRAIN_CATEGORY
            or terrain_category > MAX_TERRAIN_CATEGORY
        ):
            raise ValueError(
                "Terrain Category expected to be between 1.0 and 4.0. "
                + f"Received {terrain_category}"
            )

        self._terrain_category = terrain_category
        self._shielding_data = shielding_data

    @property
    def wind_region(self) -> WindRegion:
        """
        The Wind Region of the site.
        """

        return self._wind_region

    @property
    def terrain_category(self) -> float:
        """
        The Terrain Category of the site.
        """

        return self._terrain_category

    @property
    def shielding_data(self) -> list[tuple[float, float]] | None:
        """
        The Shielding Data of the site.
        """

        return self._shielding_data

    def v_r(
        self,
        *,
        return_period: float,
        ignore_f_x: bool = False,
        version: StandardVersion = StandardVersion.AS1170_2_2021,
    ) -> float:
        """
        Calculate the regional windspeed V_R based on the region and return period.

        Parameters
        ----------
        return_period : float
            The Average Recurrence Interval (ARI) of the windspeed.
        ignore_f_x : bool, default=False
            Whether to ignore the cyclonic region factor M_c (2021 standard),
            or the uncertainty factor F_C/F_D (older standards).
        version : StandardVersion, default=StandardVersion.AS1170_2_2021
            The version of the standard to use.

        Returns
        -------
        float
        """

        return s3_2_v_r(
            wind_region=self.wind_region,
            return_period=return_period,
            ignore_m_c=ignore_f_x,
            version=version,
        )

    def m_d_exact(
        self,
        direction: str | Real | CardinalDirection,
        version: StandardVersion = StandardVersion.AS1170_2_2021,
    ) -> M_d:
        """
        Return the wind direction multiplier wind direction.

        Notes
        -----
        Does not consider m_d in +/-45deg as required by AS1170.2 S2.3 - this should
        be considered by the user of this method.

        Parameters
        ----------
        direction : float or CardinalDirection
            The direction as an angle between 0 and 360 degrees, where:
            - 0/360 = Wind from North
            - 90 = Wind from East
            - 180 = Wind from South
            - 270 = Wind from West

            Can also be any of the CardinalDirections including "ANY" to indicate the
            wind direction is to be ignored.
        version : StandardVersion, default=StandardVersion.AS1170_2_2021
            The version of the standard to use.

        Returns
        -------
        MD
            The direction multipliers
        """

        return s3_3_m_d_exact(
            direction=direction, wind_region=self.wind_region, version=version
        )

    def m_d_des(
        self,
        direction: str | Real | CardinalDirection,
        version: StandardVersion = StandardVersion.AS1170_2_2021,
        tolerance: float = 45.0,
    ) -> M_d:
        """
        Determine the design value of the direction factor M_d,
        within +/- tolderance of direction.

        Parameters
        ----------
        direction : str | Real | CardinalDirection
            The wind direction
        version : StandardVersion
            The standard to design for.
        tolerance : float
            The tolerance in the wind angle.
            Typically 45degrees.

        Returns
        -------
        MD
            The direction multipliers
        """

        return s3_3_m_d_des(
            wind_region=self.wind_region,
            direction=direction,
            version=version,
            tolerance=tolerance,
        )

    def m_z_cat(self, z, version: StandardVersion = StandardVersion.AS1170_2_2021):
        """
        Determine the basic terrain category M_zcat at a given height and terrain.

        Notes
        -----
        Does not do any averaging for changing terrain in the windward direction.

        Parameters
        ----------
        z : float
            The height at which the windspeed is being assessed.
        version : StandardVersion, default=StandardVersion.AS1170_2_2021
            The version of the standard to use.

        Returns
        -------
        float
        """

        return s4_2_2_m_zcat_basic(
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
        direction: str | Real | CardinalDirection,
        z: float | None,
        ignore_f_x: bool = False,
        version: StandardVersion = StandardVersion.AS1170_2_2021,
        use_exact_m_d: bool = False,
        tolerance: float = 45.0,
    ) -> tuple[float, float]:
        """
        Calculate the site design velocity v_sit_beta for a WindSite.

        Parameters
        ----------
        return_period : float
            The design return period.
        direction : str | float | CardinalDirection
            The angle the wind is blowing from.
        z : float | None
            The height the wind is blowing from.
            If None, uses 10m
        ignore_f_x : bool
            Ignore the climate change multiplier.
        version : StandardVersion
            The version of the standard to design for.
        use_exact_m_d : bool
            If true use m_d for the exact direction, not the 90deg or 45deg sectors
            considered in AS1170.2.
        tolerance : float
            The tolerance in the wind angle.
            Typically 45degrees.

        Returns
        -------
        tuple[float, float]
            The design velocity for the structure and the cladding.
        """

        if z is None:
            z = 10.0

        v_r = self.v_r(
            return_period=return_period, ignore_f_x=ignore_f_x, version=version
        )
        if use_exact_m_d:
            m_d = self.m_d_exact(direction=direction, version=version)
        else:
            m_d = self.m_d_des(
                direction=direction, version=version, tolerance=tolerance
            )

        m_zcat = self.m_z_cat(z=z, version=version)
        m_s = self.m_s()
        m_t = self.m_t()
        m_lee = self.m_lee()

        base_v = v_r * m_zcat * m_s * m_t * m_lee

        return base_v * m_d.struct, base_v * m_d.cladding

    def __repr__(self):
        return (
            f"{type(self).__name__}(wind_region={self.wind_region!r}, "
            f"terrain_category={self.terrain_category}, "
            f"shielding_data={self.shielding_data!r})"
        )


def q_basic(v: float, *, rho_air: float = 1.2):
    """
    Calculate the basic wind pressure caused by a certain wind velocity.

    Notes
    -----
    - Not named consistently with the standard. This function will be called so often
    that having to remember the Standard's clause will cause issues. Also, the function
    is based on basic wind pressure theory, and is not specific to AS1170.2.

    Parameters
    ----------
    v : float
        The wind velocity. Must be in m/s.
    rho_air : float, default=1.2
        The density of air. Must be in kg / m³.

    Returns
    -------
    float
    The basic wind pressure in kPa.
    """

    return 0.5 * rho_air * v**2 / 1000.0


def s3_2_v_r_no_f_x(*, a, b, return_period, k):
    """
    Calculate the basic windspeed for a wind region. Ignores parameters F_C or F_D,
    for those use method V_R

    :param a: Windspeed parameter 'a'
    :param b: Windspeed parameter 'b'
    :param return_period: The Average Recurrence Interval (ARI) of the windspeed.
    :param k: Windspeed parameter 'k'
    """

    return a - b * return_period**-k


def s3_2_v_r(
    *,
    wind_region: WindRegion,
    return_period,
    version: StandardVersion = StandardVersion.AS1170_2_2021,
    ignore_m_c: bool = False,
) -> float:
    """
    Calculate the regional windspeed V_R based on the region and return period.

    Parameters
    ----------
    wind_region : WindRegion
        The wind region where the structure is located.
    return_period : float
        The Average Recurrence Interval (ARI) of the windspeed.
    version : StandardVersion, default=StandardVersion.AS1170_2_2021
        The version of the standard to use.
    ignore_m_c : bool, default=False
        Whether to ignore the cyclonic region factor M_c (2021 standard),
        or the uncertainty factor F_C/F_D (older standards).

    Returns
    -------
    float
    """

    init_standard_data()

    f = (
        1.0
        if ignore_m_c
        else s3_4_m_c_or_f_x(
            wind_region=wind_region, return_period=return_period, version=version
        )
    )

    # Filter the DataFrame based on 'standard' and 'region' columns
    filtered_df = STANDARD_DATA["region_windspeed_parameters"].filter(
        (pl.col("standard") == version) & (pl.col("region") == wind_region)
    )

    # Extracting the required values from the filtered DataFrame
    a, b, k, v_min = filtered_df.select(["a", "b", "k", "v_min"]).row(0)

    return max(
        f * v_min, f * s3_2_v_r_no_f_x(a=a, b=b, return_period=return_period, k=k)
    )


def s3_3_m_d_exact(
    *,
    wind_region: str | WindRegion,
    direction: str | Real | CardinalDirection,
    version: StandardVersion = StandardVersion.AS1170_2_2021,
) -> M_d:
    """
    Return the wind direction multiplier for a given region and wind direction.

    Notes
    -----
    Does not consider m_d in +/-45deg as required by AS1170.2 S2.3 - this should
    be considered by the user of this method.

    Parameters
    ----------
    wind_region : str
        The wind region where the structure is located.
    direction : float or CardinalDirection
        The direction as an angle between 0 and 360 degrees, where:
        - 0/360 = Wind from North
        - 90 = Wind from East
        - 180 = Wind from South
        - 270 = Wind from West

        Can also be one of the CardinalDirections including "ANY" to indicate the
        wind direction is to be ignored.
    version : StandardVersion, default=StandardVersion.AS1170_2_2021
        The version of the standard to use.

    Returns
    -------
    MD
        The direction multiplier
    """

    # first load some required data
    init_standard_data()

    if isinstance(direction, str):
        direction = direction.upper()

    wind_direction_defs = STANDARD_DATA["wind_direction_definitions"]

    filtered_df = STANDARD_DATA["region_direction_parameters"].filter(
        (pl.col("wind_region") == wind_region) & (pl.col("standard") == version)
    )

    f_not_clad = filtered_df.filter(pl.col("direction") == "F_not_clad")["m_d"][0]

    # next bail out early if the direction doesn't matter
    if direction == "ANY":
        m_d_clad = filtered_df.filter(pl.col("direction") == direction)["m_d"][0]
        m_d = m_d_clad * f_not_clad
        return M_d(struct=m_d, cladding=m_d_clad)

    # now throw away unneeded rows from the dataframe.
    filtered_df = filtered_df.filter(
        (pl.col("direction") != "ANY") & (pl.col("direction") != "F_not_clad")
    )

    # next if the user has provided a text direction (e.g. "NW") get the angle value.
    if str(direction) in wind_direction_defs["direction"]:
        direction = wind_direction_defs.filter(pl.col("direction") == direction)[
            "angle"
        ][0]

    # now check that direction is within the range of 0-360
    direction = float(direction) % 360

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

    return M_d(struct=m_d, cladding=m_d_clad)


def s3_3_m_d_des(
    *,
    wind_region: str,
    direction: Real | str | CardinalDirection,
    version: StandardVersion = StandardVersion.AS1170_2_2021,
    tolerance: float = 45.0,
) -> M_d:
    """
    Determine the design value of the direction factor M_d,
    within +/- tolderance of direction.

    Parameters
    ----------
    wind_region : str
        The wind region.
    direction : float
        The wind direction
    version : StandardVersion
        The standard to design for.
    tolerance : float
        The tolerance in the wind angle.
        Typically 45degrees.

    Returns
    -------
    MD
        The direction multiplier
    """

    if isinstance(direction, (str, CardinalDirection)):
        angles = [direction, direction, direction]
    else:
        angles = [
            (direction - tolerance) % 360,
            direction % 360,
            (direction + tolerance) % 360,
        ]

    m_d_vals = [
        s3_3_m_d_exact(wind_region=wind_region, direction=a, version=version)
        for a in angles
    ]

    m_d_struct = max([m.struct for m in m_d_vals])
    m_d_clad = max([m.cladding for m in m_d_vals])

    return M_d(struct=m_d_struct, cladding=m_d_clad)


def s3_4_m_c_or_f_x(
    *,
    wind_region: WindRegion | str,
    return_period: float,
    version: StandardVersion = StandardVersion.AS1170_2_2021,
):
    """
    Calculate the climate change factor M_C
    (2021 edition of standard) or F_C / F_D (2011 edition).

    Parameters
    ----------
    wind_region : WindRegion
        The wind region.
    return_period : float
        The Average Recurrence Interval (ARI) of the windspeed.
    version : StandardVersion, default=StandardVersion.AS1170_2_2021
        The version of the standard to look up.

    Returns
    -------
    float
        The climate change factor.
    """

    filtered_df = STANDARD_DATA["region_windspeed_parameters"].filter(
        (pl.col("standard") == version) & (pl.col("region") == wind_region)
    )

    m_c, m_c_min_r = filtered_df.select(["m_c", "m_c_min_r"]).row(0)

    if return_period < m_c_min_r:
        return 1.0

    return m_c


def s4_2_2_m_zcat_basic(
    *, z, terrain_category, version: StandardVersion = StandardVersion.AS1170_2_2021
) -> float:
    """
    Determine the basic terrain category M_zcat at a given height and terrain.

    Notes
    -----
    Does not do any averaging for changing terrain in the windward direction.

    Parameters
    ----------
    z : float
        The height at which the windspeed is being assessed.
    terrain_category : float
        The terrain category, as a number between 1 and 4.
        Floats are acceptable for intermediate categories.
    version : StandardVersion, default=StandardVersion.AS1170_2_2021
        The version of the standard to use.

    Returns
    -------
    float
    """

    # first load some required data
    init_standard_data()
    terrain_height_multipliers = STANDARD_DATA["terrain_height_multipliers"]

    # get the basic data into the function as np arrays as we will be interpolating
    terrain_cats = terrain_height_multipliers.filter(pl.col("standard") == version)[
        "terrain_cat"
    ].unique()

    min_cat = min(terrain_cats)
    max_cat = max(terrain_cats)

    if terrain_category < min_cat or terrain_category > max_cat:
        raise ValueError(
            f"Terrain Category {terrain_category} is outside the range "
            + f"{min_cat} to {max_cat}"
        )

    # load the M_zcat data for all terrain types
    m_z_cat_data = terrain_height_multipliers.filter(
        pl.col("standard") == version
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


def s4_2_3_m_zcat_ave():
    """
    Calculate the terrain height factor M_zcat for the averaged case.

    :return:
    """

    # TODO: Implement M_zcat_ave
    raise NotImplementedError()


def s4_3_2_shielding_parameter(
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


def s4_3_1_m_s(shielding_parameter):
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


def c_pi(
    *,
    wall_type: WallType,
    area_ratio: float,
    wind_region: WindRegion,
    governing_face: FaceType,
    c_pe: float,
    k_a: float = 1.0,
    k_l: float = 1.0,
    open_area: float = 0.0,
    volume: float = 0.0,
    version: StandardVersion = StandardVersion.AS1170_2_2021,
) -> tuple[float, float]:
    """
    Calculate the internal pressure coefficient.

    Parameters
    ----------
    wall_type : WallType
        The type of wall with the opening/s.
    area_ratio : float
        The area ratio of the governing opening/s to all other openings.
    wind_region : WindRegion
        The wind region of the structure.
    governing_face : FaceType
        The face that the opening is on.
    c_pe : float
        The external pressure coefficient at the governing face.
    k_a : float, default=1.0
        The area reduction factor for the governing opening.
    k_l : float, default=1.0
        The local pressure coefficient for the governing opening.
    open_area : float, default=0.0
        The open area of the governing opening.
    volume : float, default=0.0
        The volume of the enclosed space.
    version : StandardVersion, default=StandardVersion.AS1170_2_2021
        The version of the standard to use.

    Returns
    -------
    tuple[float, float]
    Values of (c_pi_min, c_pi_max) with the opening on the given face.
    """

    init_standard_data()

    if wall_type == WallType.OPEN:
        return t5_1_b_c_pi_open(
            area_ratio=area_ratio,
            wind_region=wind_region,
            governing_face=governing_face,
            c_pe=c_pe,
            k_a=k_a,
            k_l=k_l,
            open_area=open_area,
            volume=volume,
            version=version,
        )

    return c_pi_other()


def t5_1_b_c_pi_open(
    *,
    area_ratio: float,
    wind_region: WindRegion,
    governing_face: FaceType,
    c_pe: float,
    k_a: float = 1.0,
    k_l: float = 1.0,
    open_area: float = 0.0,
    volume: float = 0.0,
    version: StandardVersion = StandardVersion.AS1170_2_2021,
) -> tuple[float, float]:
    """
    Calculate the internal pressure coefficient of a building with an opening.

    Parameters
    ----------
    area_ratio : float
        The area ratio of the governing opening/s to all other openings.
    wind_region : WindRegion
        The wind region of the structure.
    governing_face : FaceType
        The face that the opening is on.
    c_pe : float
        The external pressure coefficient at the governing face.
    k_a : float, default=1.0
        The area reduction factor for the governing opening.
    k_l : float, default=1.0
        The local pressure coefficient for the governing opening.
    open_area : float, default=0.0
        The open area of the governing opening.
    volume : float, default=0.0
        The volume of the enclosed space.
    version : StandardVersion, default=StandardVersion.AS1170_2_2021
        The version of the standard to use.

    Returns
    -------
    tuple[float, float]
    Values of (c_pi_min, c_pi_max) with the opening on the given face.
    """

    init_standard_data()

    c_pi_data = STANDARD_DATA["cpi_t5b"].filter(pl.col("version") == version)
    c_pi_data = c_pi_data.filter(pl.col("face") == governing_face)

    c_pi_data = c_pi_data.with_columns(
        (
            pl.when(pl.col("consider_c_pe"))
            .then(pl.col("min_factor") * c_pe * k_a * k_l)
            .otherwise(pl.col("min_factor"))
        ).alias("min_factor")
    )
    c_pi_data = c_pi_data.with_columns(
        (
            pl.when(pl.col("consider_c_pe"))
            .then(pl.col("max_factor") * c_pe * k_a * k_l)
            .otherwise(pl.col("max_factor"))
        ).alias("max_factor")
    )
    k_v_val = s5_3_4_k_v(open_area=open_area, volume=volume)

    c_pi_data = c_pi_data.with_columns(
        (
            pl.when(pl.col("area_ratio") >= 6.0)  # noqa: PLR2004
            .then(pl.col("min_factor") * k_v_val)
            .otherwise(pl.col("min_factor"))
        ).alias("min_factor")
    )
    c_pi_data = c_pi_data.with_columns(
        (
            pl.when(pl.col("area_ratio") >= 6.0)  # noqa: PLR2004
            .then(pl.col("max_factor") * k_v_val)
            .otherwise(pl.col("max_factor"))
        ).alias("max_factor")
    )

    area_ratios = np.asarray(c_pi_data["area_ratio"])
    min_factor = np.asarray(c_pi_data["min_factor"])
    max_factor = np.asarray(c_pi_data["max_factor"])

    if wind_region in [WindRegion.B2, WindRegion.C, WindRegion.D]:
        area_ratio = max(area_ratio, 2.0)

    min_cpi = np.interp(area_ratio, area_ratios, min_factor)
    max_cpi = np.interp(area_ratio, area_ratios, max_factor)

    return min_cpi, max_cpi


def c_pi_other():
    raise NotImplementedError()


def s5_3_4_k_v(*, open_area, volume):
    """
    Calculate the volume coefficient K_v as per AS1170.2 Section 5.3.4.

    :param open_area: The open area on the critical face.
    :param volume: The volume of the enclosed space.
    """

    alpha = 100 * (open_area ** (3 / 2)) / volume

    if alpha < 0.09:  # noqa: PLR2004
        return 0.85

    if alpha > 3:  # noqa: PLR2004
        return 1.085

    return 1.01 + 0.15 * log10(alpha)


def s5_4_c_pe_us(*, h: float, h_us: float) -> C_pe:
    """
    Calculate the pressure coefficient on the U/S of a building as per AS1170.2 Section 5.4.

    Parameters
    ----------
    h : float
        The building reference height, in m.
    h_us : float
        The height to the U/S of the building.

    Returns
    -------
    The pressure coefficients.
    """

    x = np.asarray([0.0, 1 / 3])
    y_min = np.asarray([0.0, -0.60])
    y_max = np.asarray([0.0, 0.80])

    c_pe_min = np.interp(h_us / h, x, y_min)
    c_pe_max = np.interp(h_us / h, x, y_max)

    return C_pe(c_pe_min=float(c_pe_min), c_pe_max=float(c_pe_max))


def t5_2b_c_pe_l(
    *,
    roof_pitch: float,
    d_b_ratio: float | tuple[float, float],
    roof_type: RoofType,
    version: StandardVersion = StandardVersion.AS1170_2_2021,
) -> C_pe:
    """
    Calculate the external pressure coefficient for a leeward wall of a building.

    Parameters
    ----------
    roof_pitch : float
        The pitch of the roof. In degrees.
    d_b_ratio : float | tuple[float, float]
        The depth to breadth of the building relative to the wind.
        Either the ratio, or a tuple of the form (depth, breadth)
    roof_type : RoofType
        The type of roof. Note that this is from the direction the wind is blowing.
    version : StandardVersion, default=StandardVersion.AS1170_2_2021
        The version of the standard to use.

    Returns
    -------
    C_pe
        The external pressure coefficients.
    """

    init_standard_data()

    c_pe_data = STANDARD_DATA["cpe_t5_2b"].filter(pl.col("version") == version)
    c_pe_data = c_pe_data.filter(pl.col("hip_or_gable") == roof_type)
    c_pe_data = c_pe_data["roof_pitch", "d_b_ratio", "c_pe"].pivot(
        "roof_pitch", index="d_b_ratio"
    )

    x = np.float64(c_pe_data.columns[1:])
    y = c_pe_data.to_numpy()[:, :1].flatten()
    z = c_pe_data.to_numpy()[:, 1:].T

    interp = RegularGridInterpolator((x, y), z, bounds_error=True)

    if isinstance(d_b_ratio, tuple):
        d_b_ratio = d_b_ratio[0] / d_b_ratio[1]

    c_pe = float(interp((roof_pitch, d_b_ratio)))

    return C_pe(c_pe_min=c_pe, c_pe_max=c_pe)


def t5_2c_c_pe_s(
    *, h_ref, d_edge, version: StandardVersion = StandardVersion.AS1170_2_2021
) -> C_pe:
    """
    Calculate the external pressure coefficient for a side-wall of a building

    Parameters
    ----------
    h_ref : float
        The reference height of the building
    d_edge : float
        The distance of the point under consideration from the windward end of the building.
    version : StandardVersion, default=StandardVersion.AS1170_2_2021
        The version of the standard to use.

    Returns
    -------
    tuple[float, float]
    The external pressure coefficient. Two copies are returned for consistency with the other c_pe functions.
    """

    init_standard_data()

    c_pe_data = STANDARD_DATA["cpe_t5_2c"].filter(pl.col("version") == version)

    distance_ratio = d_edge / h_ref

    distances = np.asarray(c_pe_data["distance_edge"])
    c_pe_vals = np.asarray(c_pe_data["c_pe"])

    c_pe = np.interp(distance_ratio, distances, c_pe_vals)

    return C_pe(c_pe_min=c_pe, c_pe_max=c_pe)


def t5_3ab_c_pe_r(
    *,
    h_ref: float,
    d_ref: float,
    alpha: float,
    d_edge: float,
) -> C_pe:
    """
    Calculate the external pressure coefficient for a roof as per Table 5.3a and 5.3b of AS1170.2.

    Notes
    -----
    - This is for flat roofs or the upwind slope only.

    Parameters
    ----------
    h_ref : float
        The reference height of the roof.
    d_ref : float
        The reference depth into the wind of the roof.
    alpha : float
        The slope of the roof. In degrees.
    d_edge : float
        The distance of the point under consideration from the windward end of the building.

    Returns
    -------
    C_pe
    The external pressure coefficients.
    """

    init_standard_data()

    c_pe_data = STANDARD_DATA["cpe_t5_3"]

    loc_ratio = d_edge / h_ref
    h_d_ratio = h_ref / d_ref

    # get unique values for each of the first data values.
    alpha_vals = c_pe_data["alpha"].unique().to_numpy()
    distance_vals = c_pe_data["distance_h"].unique().to_numpy()
    h_d_vals = c_pe_data["h_d_ratio"].unique().to_numpy()

    # map values to indices in the result in numpy array
    alpha_map = {v: i for i, v in enumerate(alpha_vals)}
    distance_map = {v: i for i, v in enumerate(distance_vals)}
    h_d_map = {v: i for i, v in enumerate(h_d_vals)}

    # the shape of the array:
    shape = (len(alpha_vals), len(distance_vals), len(h_d_vals), 2)

    # create an empty numpy array:
    array = np.full(shape, np.nan)

    # populate the array:
    for row in c_pe_data.iter_rows():
        i = alpha_map[row[0]]
        j = distance_map[row[1]]
        k = h_d_map[row[2]]

        array[i, j, k, 0] = row[3]
        array[i, j, k, 1] = row[4]

    interp = RegularGridInterpolator((alpha_vals, distance_vals, h_d_vals), array)

    c_pe_vals = interp((alpha, loc_ratio, h_d_ratio))

    return C_pe(c_pe_min=c_pe_vals[0], c_pe_max=c_pe_vals[1])


def t5_4_k_a(
    area: float,
    face_type: FaceType,
    z: float | None = None,
    version: StandardVersion = StandardVersion.AS1170_2_2021,
):
    """
    Calculate the area reduction factor K_a as per AS1170.2

    Parameters
    ----------
    area : float
        The area of the face.
    face_type : FaceType
        The type of face.
    z : float | None, default=None
        The height of the point under consideration.
    version : StandardVersion, default=StandardVersion.AS1170_2_2021
        The version of the standard to use.

    Returns
    -------
    float
        The area reduction factor K_a.
    """

    init_standard_data()

    k_a_data = STANDARD_DATA["k_a"]

    if version not in k_a_data["version"].unique():
        return 1.0

    k_a_data = k_a_data.filter(pl.col("version") == version)

    if face_type not in k_a_data["face_type"].unique():
        return 1.0

    k_a_data = k_a_data.filter(pl.col("face_type") == face_type)

    max_z = max(k_a_data["h_limit"])

    if z is None:
        z = 0.0

    if z > max_z:
        return 1.0

    area_vals = np.asarray(k_a_data["area"])
    k_a_vals = np.asarray(k_a_data["k_a"])

    return np.interp(area, area_vals, k_a_vals)


def a5_2_1_c_pb(
    *, theta: float, c: float, b: float, use_radians: bool = False
) -> float:
    """
    Calculate the coefficient for normal pressure on a circular bin or tank wall.

    Parameters
    ----------
    theta : float
        The angle of the point under consideration. 0degrees is the windward location.
    c : float
        The height of the tank or bin.
    b : float
        The diameter of the tank or bin.
    use_radians : bool, default=False
        Is the angle in radians or not?
    """

    if c / b < 0.25 or c / b > 4.0:  # noqa: PLR2004
        raise ValueError(
            f"Invalid c / b ratio - ratio must satisfy 0.25 < c / b < 4.0. Current ratio is {c / b:.3f}"
        )

    if not use_radians:
        theta = radians(theta)

    c_p1 = (
        -0.5
        + 0.4 * cos(theta)
        + 0.8 * cos(2 * theta)
        + 0.3 * cos(3 * theta)
        - 0.1 * cos(4 * theta)
        - 0.05 * cos(5 * theta)
    )

    k_b = 1.0 if c_p1 >= -0.15 else 1.0 - 0.55 * (c_p1 + 0.15) * log10(c / b)  # noqa: PLR2004

    return k_b * c_p1


def k_ar(*, length: float, width: float) -> float:
    """
    Calculate the aspect ratio correction factor K_ar as per AS1170.2 Appendix C

    Parameters
    ----------
    length : float
        The length of the section into the wind.
    width : float
        The width of the section across the wind.

    Returns
    -------
    float
        The aspect ratio correction factor K_ar.
    """

    init_standard_data()

    k_ar_data = STANDARD_DATA["app_c_k_ar"]

    l_b_ratio = length / width

    return np.interp(l_b_ratio, k_ar_data["l_b_ratio"], k_ar_data["k_ar"])


def c_fig_rect_prism(d: float, b: float, theta: float = 0.0) -> tuple[float, float]:
    """
    Calculate the external pressure coefficient for a rectangular prism as per
    AS1170.2 Appendix C Figure C.2

    Parameters
    ----------
    d : float
        The depth of the section into the wind. Units to match b.
    b : float
        The width of the section across the wind. Consistent units with d.
    theta : float
        The angle of the wind relative to the section. In degrees.

    Returns
    -------
    tuple[float, float]
    The external pressure coefficient (C_fx and C_fy)
    """

    init_standard_data()

    c_fig_data = STANDARD_DATA["app_c_fig_c2"]

    if theta < 0.0 or theta > 45:  # noqa: PLR2004
        raise ValueError("Theta must be between 0 and 45 degrees")

    d_b_ratio = d / b

    data_theta_0 = c_fig_data.filter(pl.col("theta") == 0.0)
    data_theta_45 = c_fig_data.filter(pl.col("theta") == 45.0)  # noqa: PLR2004

    d_b_ratios = np.asarray(data_theta_0["d_b_ratio"])

    f_x_0_vals = np.asarray(data_theta_0["f_x"])
    f_x_45_vals = np.asarray(data_theta_45["f_x"])

    f_y_0_vals = np.asarray(data_theta_0["f_y"])
    f_y_45_vals = np.asarray(data_theta_45["f_y"])

    f_x_0 = np.interp(d_b_ratio, d_b_ratios, f_x_0_vals)
    f_x_45 = np.interp(d_b_ratio, d_b_ratios, f_x_45_vals)

    f_y_0 = np.interp(d_b_ratio, d_b_ratios, f_y_0_vals)
    f_y_45 = np.interp(d_b_ratio, d_b_ratios, f_y_45_vals)

    f_x = np.interp(theta, [0.0, 45.0], [f_x_0, f_x_45])
    f_y = np.interp(theta, [0.0, 45.0], [f_y_0, f_y_45])

    return f_x, f_y
