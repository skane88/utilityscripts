"""
File to contain some basic AS1170.2 helper methods for working with wind loads
"""

from __future__ import annotations

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


class FaceType(StrEnum):
    WINDWARD = "windward"
    LEEWARD = "leeward"
    SIDE = "side"
    ROOF = "roof"


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
            "cpe_t5_2c",
            "k_a",
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

    def __init__(self, wind_region: str, terrain_category: float, shielding_data=None):
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
        version : str, default="2021"
            The version of the standard to use.

        Returns
        -------
        float
        """

        return v_r(
            wind_region=self.wind_region,
            return_period=return_period,
            ignore_m_c=ignore_f_x,
            version=version,
        )

    def m_d(self, direction: float | str, version: str = "2021") -> tuple[float, float]:
        """
        Return the wind direction multiplier wind direction.

        Notes
        -----
        Does not consider m_d in +/-45deg as required by AS1170.2 S2.3 - this should
        be considered by the user of this method.

        Parameters
        ----------
        wind_region : str
            The wind region where the structure is located.
        direction : float or str
            The direction as an angle between 0 and 360 degrees, where:
            - 0/360 = Wind from North
            - 90 = Wind from East
            - 180 = Wind from South
            - 270 = Wind from West

            Can also be "ANY" to return the any direction value, or cardinal
            values (e.g. "N", "NE", "E", ..., "NW").
        version : str, default="2021"
            The version of the standard to use.

        Returns
        -------
        tuple[float, float]
            The direction multiplier as (M_d, M_d_cladding)
        """

        return m_d_exact(
            direction=direction, wind_region=self.wind_region, version=version
        )

    def m_d_des(
        self, direction: float | str, version: str = "2021", tolerance: float = 45.0
    ):
        """
        Determine the design value of the direction factor M_d,
        within +/- tolderance of direction.

        Parameters
        ----------
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

        return m_d_des(
            wind_region=self.wind_region,
            direction=direction,
            version=version,
            tolerance=tolerance,
        )

    def m_z_cat(self, z, version: str = "2021"):
        """
        Determine the basic terrain category M_zcat at a given height and terrain.

        Notes
        -----
        Does not do any averaging for changing terrain in the windward direction.

        Parameters
        ----------
        z : float
            The height at which the windspeed is being assessed.
        version : str, default="2021"
            The version of the standard to use.

        Returns
        -------
        float
        """

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
        use_exact_m_d: bool = False,
        tolerance: float = 45.0,
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
        use_exact_m_d : bool
            If true use m_d for the exact direction, not the 90deg or 45deg sectors
            considered in AS1170.2.
        tolerance : float
            The tolerance in the wind angle.
            Typically 45degrees.

        Returns
        -------

        """

        if z is None:
            z = 10.0

        v_r = self.v_r(
            return_period=return_period, ignore_f_x=ignore_f_x, version=version
        )
        if use_exact_m_d:
            m_d = self.m_d(direction=direction, version=version)
        else:
            m_d = self.m_d_des(
                direction=direction, version=version, tolerance=tolerance
            )

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

    def open_on_other_faces(self, face: int):
        """
        Calculate the total open area on other faces

        Parameters
        ----------
        face : int
            The face to ignore.

        Returns
        -------
        float
        """
        return sum([area for f, area in self.openings if f != face])

    def area_ratio(self, face):
        """
        Calculate the ratio of open area on a face to the sum of all the other openings.

        Parameters
        ----------
        face : int
            The face to get the area ratio for.

        Returns
        -------
        float
        """
        return self.open_area_on_face(face=face) / self.open_on_other_faces(face=face)

    @property
    def area_ratios(self) -> list[float]:
        """
        The area ratios of openings on all faces.

        Returns
        -------
        list[float]
        Area ratios list of the format:
        [area_ratio_0, ..., area_ratio_4]
        """
        return [self.area_ratio(face=f) for f in range(0, 5)]

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

    def is_gable(self, face: int):
        """
        Is the face a hip or a gable?

        Parameters
        ----------
        face : the face to check

        Returns
        -------
        bool
        """

        gables = [0, 2]

        return face in gables

    def roof_pitch_for_face(self, face) -> float:
        """
        What is the effective roof pitch for wind blowing on a face?

        Parameters
        ----------
        face : int
            The face to check.

        Returns
        -------
        float
        """

        if self.is_gable(face):
            return 0.0
        return self.roof_pitch


class OpenStructure:
    """
    Class to represent an open structure and determine wind loads as per AS1170.2
    Appendix C.
    """

    def __init__(
        self,
        *,
        frame_h: float,
        frame_l: float,
        frame_s: float,
    ):
        """
        Initialise an OpenStructure object.

        Notes
        -----
        The created OpenStructure object does not have any members defined yet.
        Use the add_member method to add members to the structure.

        Parameters
        ----------
        frame_h : float
            The height of the frame into the wind. In m.
        frame_l : float
            The length of the frame. In m.
        frame_s : float
            The spacing of the frames. In m.
        """

        self._member_data = pl.DataFrame(
            {
                "name": pl.Series([], dtype=pl.Utf8),
                "depth": pl.Series([], dtype=pl.Float64),
                "length": pl.Series([], dtype=pl.Float64),
                "reference_height": pl.Series([], dtype=pl.Float64),
                "drag_coefficient": pl.Series([], dtype=pl.Float64),
                "no_unshielded": pl.Series([], dtype=pl.Int64),
                "no_shielded": pl.Series([], dtype=pl.Int64),
                "ignore_for_area": pl.Series([], dtype=pl.Boolean),
                "circular_or_sharp": pl.Series([], dtype=pl.Utf8),
            }
        )
        self._frame_h = frame_h
        self._frame_l = frame_l
        self._frame_s = frame_s

    def _copy_with_new(self, **new_attributes) -> OpenStructure:
        """
        Function to copy the OpenStructure instance but update specific attributes.

        The returned copy is a deepcopy.
        Note that the function replaces _geometry of the new instance with None.
        The user of the instance is responsible for recreating the geometry.

        Parameters
        ----------
        new_attributes : dict
            Any attributes to update as key:value pairs.

        Returns
        -------
        OpenStructure
            A new instance of an OpenStructure with updated attributes.
        """

        new_structure = copy.deepcopy(self)

        for attr, value in new_attributes.items():
            setattr(new_structure, attr, value)

        return new_structure

    @property
    def member_data(self) -> pl.DataFrame:
        """
        The member data for the open structure.

        Returns
        -------
        pl.DataFrame

        A dataframe with the following columns:
            - name: a name for each section
            - depth: the depth of the section in m
            - length: the length of the section in m
            - reference_height: the reference height of the section in m
            - drag_coefficient: the drag coefficient for each section
            - no_unshielded: the number of unshielded sections
            - no_shielded: the number of shielded sections
            - ignore_for_area: should the sections be ignored for overall
                area calculations?
            - circular_or_sharp: are the sections circular or sharp edged?
        """

        return self._member_data

    @property
    def frame_h(self) -> float:
        return self._frame_h

    @property
    def frame_l(self) -> float:
        return self._frame_l

    @property
    def frame_s(self) -> float:
        return self._frame_s

    @property
    def projected_area(self) -> float:
        """
        The projected area of the open structure.
        """

        return self.frame_l * self.frame_h

    def add_member(
        self,
        *,
        name: str,
        depth: float,
        length: float,
        reference_height: float,
        drag_coefficient: float,
        no_unshielded: int,
        no_shielded: int,
        ignore_for_area: bool = False,
        circular_or_sharp: str = "circular",
    ) -> OpenStructure:
        """
        Add a member to the open structure.

        Notes
        -----
        The method does not update the OpenStructure in place. A new OpenStructure
        object is returned.

        Parameters
        ----------
        name : str
            The name of the member.
        depth : float
            The depth of the member.
        length : float
            The length of the member.
        reference_height : float
            The reference height of the member.
        drag_coefficient : float
            The drag coefficient for the member.
        no_unshielded : int
            The number of unshielded sections.
        no_shielded : int
            The number of shielded sections.
        ignore_for_area : bool, default=False
            Should the sections be ignored for overall area calculations?
        circular_or_sharp : str, default="circular"
            Are the sections circular or sharp edged?

        Returns
        -------
        OpenStructure
            A new instance of an OpenStructure with the updated member data.
        """

        return self._copy_with_new(
            _member_data=pl.concat(
                [
                    self._member_data,
                    pl.DataFrame(
                        {
                            "name": [name],
                            "depth": [depth],
                            "length": [length],
                            "reference_height": [reference_height],
                            "drag_coefficient": [drag_coefficient],
                            "no_unshielded": [no_unshielded],
                            "no_shielded": [no_shielded],
                            "ignore_for_area": [ignore_for_area],
                            "circular_or_sharp": [circular_or_sharp],
                        }
                    ),
                ]
            )
        )


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
) -> float:
    """
    Calculate the regional windspeed V_R based on the region and return period.

    Parameters
    ----------
    wind_region : str
        The wind region where the structure is located.
    return_period : float
        The Average Recurrence Interval (ARI) of the windspeed.
    version : str, default="2021"
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

    Notes
    -----
    Does not consider m_d in +/-45deg as required by AS1170.2 S2.3 - this should
    be considered by the user of this method.

    Parameters
    ----------
    wind_region : str
        The wind region where the structure is located.
    direction : float or str
        The direction as an angle between 0 and 360 degrees, where:
        - 0/360 = Wind from North
        - 90 = Wind from East
        - 180 = Wind from South
        - 270 = Wind from West

        Can also be "ANY" to return the any direction value, or cardinal
        values (e.g. "N", "NE", "E", ..., "NW").
    version : str, default="2021"
        The version of the standard to use.

    Returns
    -------
    tuple[float, float]
        The direction multiplier as (M_d, M_d_cladding)
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
    tolerance: float = 45.0,
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
    version : str, default="2021"
        The version of the standard to use.

    Returns
    -------
    float
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
    m_z_cat_data = terrain_height_multipliers.filter(
        pl.col("standard") == int(version)
    ).pivot_table("terrain_cat", index="height", values="m_z_cat")

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


def c_pi(
    *,
    wall_type: WallType,
    area_ratio: float,
    is_cyclonic: bool,
    governing_face: FaceType,
    c_pe: float,
    k_a: float = 1.0,
    k_l: float = 1.0,
    open_area: float = 0.0,
    volume: float = 0.0,
    version: str = "2021",
) -> tuple[float, float]:
    """
    Calculate the internal pressure coefficient.

    Parameters
    ----------
    wall_type : WallType
        The type of wall with the opening/s.
    area_ratio : float
        The area ratio of the governing opening/s to all other openings.
    is_cyclonic : bool
        Whether the structure is in a cyclonic region.
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
    version : str, default="2021"
        The version of the standard to use.

    Returns
    -------
    tuple[float, float]
    Values of (c_pi_min, c_pi_max) with the opening on the given face.
    """

    init_standard_data()

    if wall_type == WallType.OPEN:
        return c_pi_open(
            area_ratio=area_ratio,
            is_cyclonic=is_cyclonic,
            governing_face=governing_face,
            c_pe=c_pe,
            k_a=k_a,
            k_l=k_l,
            open_area=open_area,
            volume=volume,
            version=version,
        )

    return c_pi_other()


def c_pi_open(
    *,
    area_ratio: float,
    is_cyclonic: bool,
    governing_face: FaceType,
    c_pe: float,
    k_a: float = 1.0,
    k_l: float = 1.0,
    open_area: float = 0.0,
    volume: float = 0.0,
    version: str = "2021",
) -> tuple[float, float]:
    """
    Calculate the internal pressure coefficient of a building with an opening.

    Parameters
    ----------
    area_ratio : float
        The area ratio of the governing opening/s to all other openings.
    is_cyclonic : bool
        Whether the structure is in a cyclonic region.
    governing_face : FaceType
        The face that the opening is on.
    c_pe : float
        The external pressure coefficient at the governing face.
    k_a : float, default=1.0
        The area reduction factor for the governing opening.
    k_l : float, default=1.0
        The local pressure coefficient for the governing opening.

    Returns
    -------
    tuple[float, float]
    Values of (c_pi_min, c_pi_max) with the opening on the given face.
    """

    init_standard_data()

    c_pi_data = STANDARD_DATA["cpi_t5b"].filter(pl.col("version") == int(version))
    c_pi_data = c_pi_data.filter(pl.col("face") == governing_face)

    c_pi_data = c_pi_data.with_columns(
        pl.when(pl.col("c_pe"))
        .then(pl.col("min_factor") * c_pe * k_a * k_l)
        .otherwise(pl.col("min_factor"))
    )
    c_pi_data = c_pi_data.with_columns(
        pl.when(pl.col("c_pe"))
        .then(pl.col("max_factor") * c_pe * k_a * k_l)
        .otherwise(pl.col("max_factor"))
    )

    area_ratios = np.asarray(c_pi_data["area_ratio"])
    min_factor = np.asarray(c_pi_data["min_factor"])
    max_factor = np.asarray(c_pi_data["max_factor"])

    if is_cyclonic:
        area_ratio = max(area_ratio, 2.0)

    min_cpi = np.interp(area_ratio, area_ratios, min_factor)
    max_cpi = np.interp(area_ratio, area_ratios, max_factor)

    if area_ratio >= 6.0 and governing_face != FaceType.ROOF:  # noqa: PLR2004
        k_v_val = k_v(open_area=open_area, volume=volume)
    else:
        k_v_val = 1.0

    return min_cpi * k_v_val, max_cpi * k_v_val


def c_pi_other():
    raise NotImplementedError()


def q_basic(v: float, *, rho_air: float = 1.2):
    """
    Calculate the basic wind pressure caused by a certain wind velocity.

    Parameters
    ----------
    v : float
        The wind velocity. Must be in m/s.
    rho_air : float, default=1.2
        The density of air. Must be in kg / m³.

    Returns
    -------
    float
    The basic wind pressure in Pa.
    """

    return 0.5 * rho_air * v**2


def c_pe_s(*, h_ref, d_edge, version="2021") -> tuple[float, float]:
    """
    Calculate the external pressure coefficient for a side-wall of a building

    Parameters
    ----------
    h_ref : float
        The reference height of the building
    d_edge : float
        The distance of the point under consideration from the windward end of the building.
    version : str, default="2021"
        The version of the standard to use.

    Returns
    -------
    tuple[float, float]
    The external pressure coefficient. Two copies are returned for consistency with the other c_pe functions.
    """

    init_standard_data()

    c_pe_data = STANDARD_DATA["cpe_t5_2c"].filter(pl.col("version") == int(version))

    distance_ratio = d_edge / h_ref

    distances = np.asarray(c_pe_data["distance_edge"])
    c_pe_vals = np.asarray(c_pe_data["c_pe"])

    c_pe = np.interp(distance_ratio, distances, c_pe_vals)

    return c_pe, c_pe


def pipe_wind_loads(cd, qz, d_max, d_ave, n_pipes):
    """
    Calculate wind loads on pipes in pipe racks.

    Notes
    -----
    Based on Oil Search design criteria 100-SPE-K-0001.

    Parameters
    ----------
    cd : float
        The drag coefficient for the largest pipe in the group.
    qz : float
        The design wind pressure.
    d_max : float
        The maximum pipe diameter.
    d_ave : float
        The average pipe diameter.
    n_pipes : int
        The number of pipes in the tray.
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


def k_a(
    area: float, face_type: FaceType, z: float | None = None, version: str = "2021"
):
    init_standard_data()

    k_a_data = STANDARD_DATA["k_a"]

    k_a_data = k_a_data.filter(pl.col("version") == int(version))
    k_a_data = k_a_data.filter(pl.col("face_type") == face_type)

    max_z = max(k_a_data["h_limit"])

    if z is None:
        z = 0.0

    if z > max_z:
        return 1.0

    area_vals = np.asarray(k_a_data["area"])
    k_a_vals = np.asarray(k_a_data["k_a"])

    return np.interp(area, area_vals, k_a_vals)


def c_fig_rect_prism(d, b, theta):
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
