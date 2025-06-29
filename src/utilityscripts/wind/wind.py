"""
Contains some calculations for wind loads.
"""

from __future__ import annotations

from copy import deepcopy
from math import radians, tan

import numpy as np
import polars as pl

from utilityscripts.wind.as1170_2 import (
    SectionType,
    StandardVersion,
    WindRegion,
    WindSite,
    k_ar,
    q_basic,
    s3_3_m_d_des,
    s3_4_m_c_or_f_x,
    s4_2_2_m_zcat_basic,
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
        version: StandardVersion = StandardVersion.AS1170_2_2021,
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
        self._version = version

    @property
    def wind_site(self) -> WindSite:
        """
        The wind site object used to determine windspeeds etc. for the building.
        """

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

    @property
    def version(self) -> StandardVersion:
        return self._version

    def add_opening(self, opening: tuple[int, float]) -> "SimpleBuilding":
        new_building = deepcopy(self)
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

    def m_d(
        self, version: StandardVersion = StandardVersion.AS1170_2_2021
    ) -> list[tuple[float, float]]:
        """
        Calculate the value of M_d for each face.

        Returns
        -------
        A list of the form [M_d0, M_d1, M_d2, M_d3]
        """

        return [
            s3_3_m_d_des(
                wind_region=self.wind_site.wind_region, direction=a, version=version
            )
            for a in self.design_angles
        ]

    def m_d_on_face(
        self, face: int, version: StandardVersion = StandardVersion.AS1170_2_2021
    ) -> tuple[float, float]:
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
        return s3_3_m_d_des(
            wind_region=self.wind_site.wind_region,
            direction=self.face_angle(face=face),
            version=version,
        )

    def m_c_or_f_x(
        self,
        wind_region: WindRegion | str,
        return_period: float,
        version: StandardVersion = StandardVersion.AS1170_2_2021,
    ):
        """
        Get the value of M_c or F_C / F_D as appropriate.

        Parameters
        ----------
        wind_region : str
            The wind region.
        return_period : float
            The return period.
        version : StandardVersion
            The version of the standard to check.

        Returns
        -------
        float
        """
        return s3_4_m_c_or_f_x(
            wind_region=wind_region, return_period=return_period, version=version
        )

    def v_sit_beta(
        self,
        return_period: float,
        z: float | None = None,
        version: StandardVersion = StandardVersion.AS1170_2_2021,
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
        version : StandardVersion
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
        self,
        face: int,
        return_period: float,
        z: float | None = None,
        version: StandardVersion = StandardVersion.AS1170_2_2021,
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
        version : StandardVersion
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
        v_r: float,
        terrain_category: float,
        m_s: float = 1.0,
        m_t: float = 1.0,
        m_d: float = 1.0,
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
            This is not the height above ground, just the relative height from the
            lowest level of the frame to the highest.
        frame_l : float
            The length of the frame. In m.
        frame_s : float
            The spacing of the frames. In m.
        v_r : float
            The design windspeed at a given return period. In m/s.
        terrain_category : float
            The terrain category of the site. A float between 1.0 and 4.0. Intermediate
            values are linearly interpolated between the values given in AS1170.2 for
            the integer values.
        m_s : float
            The shielding factor for the structure.
        m_t : float
            The terrain factor for the structure.
        m_d : float
            The wind direction factor for the structure.
        """

        self._member_data = pl.DataFrame(
            {
                "component_id": pl.Series([], dtype=pl.Utf8),
                "name": pl.Series([], dtype=pl.Utf8),
                "master_component": pl.Series([], dtype=pl.Utf8),
                "comments": pl.Series([], dtype=pl.Utf8),
                "depth": pl.Series([], dtype=pl.Float64),
                "length": pl.Series([], dtype=pl.Float64),
                "reference_height": pl.Series([], dtype=pl.Float64),
                "drag_coefficient": pl.Series([], dtype=pl.Float64),
                "no_per_frame": pl.Series([], dtype=pl.Int64),
                "no_unshielded_frames": pl.Series([], dtype=pl.Int64),
                "no_shielded_frames": pl.Series([], dtype=pl.Int64),
                "include_in_solidity": pl.Series([], dtype=pl.Boolean),
                "circular_or_sharp": pl.Series([], dtype=pl.Utf8),
                "inclination": pl.Series([], dtype=pl.Float64),
                "k_sh": pl.Series([], dtype=pl.Float64),
            }
        )
        self._frame_h = frame_h
        self._frame_l = frame_l
        self._frame_s = frame_s
        self._v_r = v_r
        self._terrain_category = terrain_category
        self._m_s = m_s
        self._m_t = m_t
        self._m_d = m_d

        self._results = deepcopy(self._member_data)

    def _copy(self) -> OpenStructure:
        """
        Function to copy the OpenStructure instance.

        The returned copy is a deepcopy.

        Returns
        -------
        OpenStructure
            A new instance of an OpenStructure
        """

        return deepcopy(self)

    @property
    def member_data(self) -> pl.DataFrame:
        """
        The member data for the open structure.

        Returns
        -------
        pl.DataFrame

        A dataframe with the following columns:
            - id: a unique identifier for each section
            - name: a name for each section
            - master_component: a master component (if any) the member is part of.
                Included for sorting, filtering purposes.
            - comments: any text comments to attach.
            - depth: the depth of the section in m
            - length: the length of the section in m
            - reference_height: the reference height of the section in m
            - drag_coefficient: the drag coefficient for each section
            - no_per_frame: the number of sections per frame
            - no_unshielded_frames: the number of unshielded frames
            - no_shielded_frames: the number of shielded frames
            - include_in_solidity: should the sections be included in overall
                area solidity calculations?
            - circular_or_sharp: are the sections circular or sharp edged?
            - inclination: the inclination of the section to the wind, in degrees
            - k_sh: the shielding factor for the section
        """

        return self._member_data

    @property
    def frame_h(self) -> float:
        """
        The height of the frame into the wind. In m.
        This is not the height above ground, just the relative height from the
        lowest level of the frame to the highest.

        Returns
        -------
        float
        """

        return self._frame_h

    @property
    def frame_l(self) -> float:
        """
        The length of the frame. In m.
        """

        return self._frame_l

    @property
    def frame_s(self) -> float:
        """
        The spacing of the frames. In m.
        """

        return self._frame_s

    @property
    def v_r(self) -> float:
        """
        The design windspeed at a given return period.
        """

        return self._v_r

    @property
    def terrain_category(self) -> float:
        """
        The terrain category of the site. A float between 1.0 and 4.0. Intermediate
        values are linearly interpolated between the values given in AS1170.2 for
        the integer values.
        """

        return self._terrain_category

    @property
    def m_s(self) -> float:
        """
        The shielding factor for the structure.
        """

        return self._m_s

    @property
    def m_t(self) -> float:
        """
        The terrain factor for the structure.
        """

        return self._m_t

    @property
    def m_d(self) -> float:
        """
        The wind direction factor for the structure.
        """

        return self._m_d

    @property
    def projected_area(self) -> float:
        """
        The projected area of the open structure.
        """

        return self.frame_l * self.frame_h

    @property
    def results(self) -> pl.DataFrame:
        """
        Returns the results of the wind load calculations.

        Returns
        -------
        pl.DataFrame
            A dataframe with the member-by-member calculations.
        """

        if self._results.is_empty():
            self._calculate()

        return self._results

    def add_member(
        self,
        *,
        component_id: str,
        name: str,
        depth: float,
        length: float,
        reference_height: float,
        drag_coefficient: float,
        no_per_frame: int,
        no_unshielded_frames: int,
        no_shielded_frames: int,
        include_in_solidity: bool = True,
        circular_or_sharp: SectionType = SectionType.CIRCULAR,
        master_component: str = "",
        comments: str = "",
        inclination: float = 90.0,
        k_sh: float = 1.0,
    ) -> OpenStructure:
        """
        Add a member to the open structure.

        Notes
        -----
        The method does not update the OpenStructure in place. A new OpenStructure
        object is returned.

        Parameters
        ----------
        component_id : str
            A unique identifier for the member.
        name : str
            The name of the member.
        master_component : str
            A master component (if any) the member is part of. Included for sorting,
            filtering purposes.
        comments : str, default=""
            Any text comments to attach.
        depth : float
            The depth of the member.
        length : float
            The length of the member.
        reference_height : float
            The reference height of the member.
        drag_coefficient : float
            The drag coefficient for the member.
        no_per_frame : int
            The number of sections per frame.
        no_unshielded_frames : int
            The number of unshielded frames.
        no_shielded_frames : int
            The number of shielded frames.
        include_in_solidity : bool, default=True
            Should the sections be included in overall area solidity calculations?
        circular_or_sharp : SectionType, default=SectionType.CIRCULAR
            Are the sections circular or sharp edged?
        inclination : float, default=90.0
            The inclination of the section to the wind, in degrees
        k_sh : float, default=1.0
            The shielding factor for the section

        Returns
        -------
        OpenStructure
            A new instance of an OpenStructure with the updated member data.
        """

        new_structure = self._copy()
        new_structure._member_data = pl.concat(
            [
                self._member_data,
                pl.DataFrame(
                    {
                        "component_id": [component_id],
                        "name": [name],
                        "master_component": [master_component],
                        "comments": [comments],
                        "depth": [depth],
                        "length": [length],
                        "reference_height": [reference_height],
                        "drag_coefficient": [drag_coefficient],
                        "no_per_frame": [no_per_frame],
                        "no_unshielded_frames": [no_unshielded_frames],
                        "no_shielded_frames": [no_shielded_frames],
                        "include_in_solidity": [include_in_solidity],
                        "circular_or_sharp": [str(circular_or_sharp)],
                        "inclination": [inclination],
                        "k_sh": [k_sh],
                    }
                ),
            ]
        )
        return new_structure

    def add_members(self, members: dict[str, list] | pl.DataFrame) -> OpenStructure:
        """
        Add multiple members to the open structure.

        Parameters
        ----------
        members : dict[str, list]
            A dictionary of members to add.

            Should have the following keys:

            - component_id: a unique identifier for each section
            - name: a name for each section
            - depth: the depth of the section in m
            - length: the length of the section in m
            - reference_height: the reference height of the section in m
            - drag_coefficient: the drag coefficient for each section
            - no_per_frame: the number of sections per frame
            - no_unshielded_frames: the number of unshielded frames
            - no_shielded_frames: the number of shielded frames
            - include_in_solidity: should the sections be included in overall
                area solidity calculations?
            - circular_or_sharp: are the sections circular or sharp edged? Should be
                a list of SectionType enums or matching strings.
            - master_component: a master component (if any) the member is part of.
                Included for sorting, filtering purposes.
            - comments: any text comments to attach.

            All lists should have the same length.

            If a Polars Dataframe is provided it should have columns with the same
            names.

        Returns
        -------
        OpenStructure
            A new instance of an OpenStructure with the updated member data.
        """

        members["circular_or_sharp"] = [str(s) for s in members["circular_or_sharp"]]

        new_structure = self._copy()
        new_structure._member_data = (
            pl.concat(
                [
                    self._member_data,
                    pl.DataFrame(members),
                ]
            )
            if isinstance(members, dict)
            else pl.concat(
                [
                    self._member_data,
                    members,
                ]
            )
        )
        return new_structure

    def _calculate(self):
        """
        Calculate the resulting wind loads.
        """

        self._results = self._member_data.with_columns(
            (
                pl.col("reference_height").map_elements(
                    lambda x: s4_2_2_m_zcat_basic(
                        z=x, terrain_category=self.terrain_category
                    ),
                    return_dtype=pl.Float64,
                )
            ).alias("M_zcat")
        )

        self._results = self._results.with_columns(
            (self.v_r * self.m_s * self.m_t * self.m_d * pl.col("M_zcat")).alias(
                "V_des"
            )
        )

        self._results = self._results.with_columns(
            (
                pl.col("V_des").map_elements(
                    lambda x: q_basic(x), return_dtype=pl.Float64
                )
            ).alias("q")
        )

        self._results = self._results.with_columns(
            (pl.col("length") / pl.col("depth")).alias("aspect_ratio")
        )

        self._results = self._results.with_columns(
            (
                pl.struct(["length", "depth"]).map_elements(
                    lambda x: k_ar(length=x["length"], width=x["depth"]),
                    return_dtype=pl.Float64,
                )
            ).alias("K_ar")
        )


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
    Calculate a temporary design windspeed V_R, for structures with design lives of
    less than 1 year.

    Notes
    -----
    This is calculated based on "Design Wind Speeds for Temporary Structures"
    by Wang & Pham (AJSE Vol 12, No 2, 2012). This is implicitly accepted by the
    Australian Building Codes Board, as they reproduce the tables from this paper in
    their "Temporary Structures" Standard, 2015.

    Parameters
    ----------
    a : float
        Windspeed parameter a.
    b : float
        Windspeed parameter b.
    k : float
        Windspeed parameter k.
    r_s : float
        The reference return period for a normal structure.
        Typically this should be taken from the Building Code.
        It should not be taken from AS1170.0 Appendix F.
    t : float
        The number of reference periods in a year.

    Returns
    -------
    float
        The temporary design windspeed.
    """

    return a - b * (1 - (1 - (1 / r_s)) ** t) ** k
