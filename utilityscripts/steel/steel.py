"""
To contain some utilities for steel design
"""

from __future__ import annotations

import copy
from math import atan, cos, degrees, log, pi, radians, sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sectionproperties.analysis.section import Section
from sectionproperties.pre.geometry import Geometry
from sectionproperties.pre.library.primitive_sections import (
    rectangular_section,
    triangular_section,
)
from sectionproperties.pre.library.steel_sections import mono_i_section

from utilityscripts.section_prop import build_circle

# All the standard I & C sections as dataframes
_DATA_PATH = Path(Path(__file__).parent) / Path("data")


def steel_grade_df() -> pd.DataFrame:
    """
    Get a Pandas Dataframe with all the Australian Standard steel grades.
    """

    return pd.read_excel(_DATA_PATH / Path("steel_data.xlsx"), sheet_name="grades")


class SteelGrade:
    def __init__(
        self,
        *,
        standard: str,
        current: bool,
        form: str,
        grade: str,
        thickness: list[float] | np.ndarray,
        f_y: list[float] | np.ndarray,
        f_u: list[float] | np.ndarray,
    ):
        """

        :param standard: The standard the steel is made to.
        :param current: Is the steel a currently produced steel?
        :param form: The form of the steel (e.g. plate, section etc.).
        :param grade: The grade designation.
        :param thickness: An array of thicknesses.
            Must be sorted smallest to largest.
            Typ. should start at 0 thick, or whatever minimum thickness
            the steel is produced in.
        :param f_y: An array of yield strengths.
            Should be sorted to match thickness.
        :param f_u: An array of ultimate strengths.
            Should be sorted to match thickness.
        """

        if len(thickness) != len(f_y) or len(thickness) != len(f_u):
            raise ValueError("Lengths of thickness, f_y and f_u need to be the same.")

        self.standard = standard
        self.current = current
        self.form = form
        self.grade = grade
        self.thickness = np.asarray(thickness)
        self.f_y = np.asarray(f_y)
        self.f_u = np.asarray(f_u)

    def get_f_y(self, thickness):
        """
        Get the yield strength at a given thickness.

        :param thickness: The thickness to test.
        """

        return np.interp(x=thickness, xp=self.thickness, fp=self.f_y)

    def get_f_u(self, thickness):
        """
        Get the ultimate strength at a given thickness.

        :param thickness: The thickness to test.
        """

        return np.interp(x=thickness, xp=self.thickness, fp=self.f_u)

    def plot_grade(self, *, strength: str = "both"):
        """
        Plot the strength of the steel vs thickness.

        :param strength: Either 'f_y' or 'f_u' or 'both'
        """

        if strength != "f_y":
            plt.plot(self.thickness, self.f_u, label="f_y")

        if strength != "f_u":
            plt.plot(self.thickness, self.f_y, label="f_u")

        plt.legend()
        plt.title("Steel Strength vs Thickness")
        plt.xlabel("Thickness")
        plt.ylabel("Steel Strength")
        plt.ylim(bottom=0)
        plt.xlim(left=0)

        plt.show()

    def __repr__(self):
        return f"{type(self).__name__}: {self.standard}:{self.grade}"


def steel_grades() -> dict[str, SteelGrade]:
    """
    Build a dictionary of steel grades out of the steel data spreadsheet.
    """

    sg_df = steel_grade_df().sort_values(["standard", "grade", "t"])

    unique_grades = sg_df.drop_duplicates(subset=["standard", "grade"])

    grades = {}

    for _, standard, current, form, grade, *_ in unique_grades.itertuples():
        t = sg_df[(sg_df.standard == standard) & (sg_df.grade == grade)].t.to_numpy()
        f_y = sg_df[
            (sg_df.standard == standard) & (sg_df.grade == grade)
        ].f_y.to_numpy()
        f_u = sg_df[
            (sg_df.standard == standard) & (sg_df.grade == grade)
        ].f_u.to_numpy()

        current = current == "yes"
        grade = str(grade)

        grades[standard + ":" + grade] = SteelGrade(
            standard=standard,
            current=current,
            form=form,
            grade=grade,
            thickness=t,
            f_y=f_y,
            f_u=f_u,
        )

    return grades


def standard_grades() -> dict[str, SteelGrade]:
    """
    Return the standard steel grades for current sections.
    """

    sg = steel_grades()

    sg300 = sg["AS/NZS3679.1:300"]
    sg300_w = sg["AS/NZS3678:300"]
    c350 = sg["AS/NZS1163:C350"]

    return {
        "UB": sg300,
        "UC": sg300,
        "UBP": sg300,
        "PFC": sg300,
        "WB": sg300_w,
        "WC": sg300_w,
        "EA": sg300,
        "UA": sg300,
        "RHS": c350,
        "SHS": c350,
        "CHS": c350,
    }


class SteelSection:
    def __init__(
        self, *, section: str, current: bool, steel_grade: None | SteelGrade = None
    ):
        """

        :param section: The section name.
        :param current: Is the section a currently produced section?
        :param steel_grade: A steel grade to attach to the section, or None.
        """

        self.section = section
        self.current = current
        self._grade = steel_grade

    @property
    def grade(self):
        return self._grade

    def set_grade(self, grade: SteelGrade) -> SteelSection:
        """
        Add a steel grade to the object.

        :param grade: The SteelGrade to apply.
        :return: Returns a copy of the object.
        """

        section_copy = copy.deepcopy(self)
        section_copy._grade = grade

        return section_copy

    def __repr__(self):
        grade = "no steel assigned" if self.grade is None else repr(self.grade)

        return (
            f"{type(self).__name__}: {self.section}, "
            + f"{grade}."
            + f" Current section: {self.current}"
        )


class ISectionData(SteelSection):
    """
    A class to store I-section properties taken directly from standard tables.
    """

    def __init__(
        self,
        *,
        section: str,
        current: bool,
        designation: str,
        mass: float,
        section_shape: float,
        fabrication_type: float,
        d: float,
        b_f: float,
        t_f: float,
        t_w: float,
        r_1: float,
        w_1: float,
        r1_or_w1: float,
        a_g: float,
        i_x: float,
        z_x: float,
        s_x: float,
        r_x: float,
        i_y: float,
        z_y: float,
        s_y: float,
        r_y: float,
        j: float,
        i_w: float,
        steel_grade: None | SteelGrade = None,
        **kwargs,
    ):
        super().__init__(section=section, current=current, steel_grade=steel_grade)

        self.designation = designation
        self.mass = mass
        self.section_shape = section_shape
        self.fabrication_type = fabrication_type
        self.d = d
        self.b_f = b_f
        self.t_f = t_f
        self.t_w = t_w
        self.r_1 = r_1
        self.w_1 = w_1
        self.r1_or_w1 = r1_or_w1
        self.a_g = a_g
        self.i_x = i_x
        self.z_x = z_x
        self.s_x = s_x
        self.r_x = r_x
        self.i_y = i_y
        self.z_y = z_y
        self.s_y = s_y
        self.r_y = r_y
        self.j = j
        self.i_w = i_w

    @property
    def f_yf(self):
        """
        Get the flange yield stress from the steel grade.
        """

        if self.grade is None:
            return None

        return self.grade.get_f_y(self.t_f)

    @property
    def f_yw(self):
        """
        Get the web yield stress from the steel grade.
        """

        if self.grade is None:
            return None

        return self.grade.get_f_y(self.t_w)

    @property
    def f_uf(self):
        """
        Get the flange ultimate stress from the steel grade.
        """

        if self.grade is None:
            return None

        return self.grade.get_f_u(self.t_f)

    @property
    def f_uw(self):
        """
        Get the web ultimate stress from the steel grade.
        """

        if self.grade is None:
            return None

        return self.grade.get_f_u(self.t_w)

    @property
    def d_1(self):
        return self.d - 2 * self.t_f


class CSectionData(SteelSection):
    """
    A class to store C section properties.
    """

    def __init__(
        self,
        *,
        section: str,
        current: bool,
        designation: str,
        mass: float,
        section_shape: float,
        fabrication_type: float,
        d: float,
        b_f: float,
        t_f: float,
        t_w: float,
        r_1: float,
        a_g: float,
        i_x: float,
        z_x: float,
        s_x: float,
        r_x: float,
        i_y: float,
        z_yl: float,
        z_yr: float,
        s_y: float,
        r_y: float,
        j: float,
        i_w: float,
        steel_grade: None | SteelGrade = None,
        **kwargs,
    ):
        super().__init__(section=section, current=current, steel_grade=steel_grade)

        self.designation = designation
        self.mass = (mass,)
        self.section_shape = section_shape
        self.fabrication_type = fabrication_type
        self.d = d
        self.b_f = b_f
        self.t_f = t_f
        self.t_w = t_w
        self.r_1 = r_1
        self.a_g = a_g
        self.i_x = i_x
        self.z_x = z_x
        self.s_x = s_x
        self.r_x = r_x
        self.i_y = i_y
        self.z_yl = z_yl
        self.z_yr = z_yr
        self.s_y = s_y
        self.r_y = r_y
        self.j = j
        self.i_w = i_w

    @property
    def f_yf(self):
        """
        Get the flange yield stress from the steel grade.
        """

        if self.grade is None:
            return None

        return self.grade.get_f_y(self.t_f)

    @property
    def f_yw(self):
        """
        Get the web yield stress from the steel grade.
        """

        if self.grade is None:
            return None

        return self.grade.get_f_y(self.t_w)

    @property
    def f_uf(self):
        """
        Get the flange ultimate stress from the steel grade.
        """

        if self.grade is None:
            return None

        return self.grade.get_f_u(self.t_f)

    @property
    def f_uw(self):
        """
        Get the web ultimate stress from the steel grade.
        """

        if self.grade is None:
            return None

        return self.grade.get_f_u(self.t_w)

    @property
    def d_1(self):
        return self.d - 2 * self.t_f


class AngleSectionData(SteelSection):
    """
    Class to map to an angle section.
    """

    # TODO: fill out class.

    def __init__(
        self,
        *,
        section: str,
        current: bool,
        designation: str,
        mass: float,
        section_shape: float,
        fabrication_type: float,
        b_1: float,
        b_2: float,
        t: float,
        r_1: float,
        r_2: float,
        a_g: float,
        n_l: float,
        n_r: float,
        p_b: float,
        p_t: float,
        i_x: float,
        y_1: float,
        z_x1: float,
        y_4: float,
        z_x4: float,
        y_5: float,
        z_x5: float,
        s_x: float,
        r_x: float,
        i_y: float,
        x_5: float,
        z_y5: float,
        x_3: float,
        z_y3: float,
        x_2: float,
        z_y2: float,
        s_y: float,
        r_y: float,
        j: float,
        tan_alpha: float,
        i_n: float,
        z_nb: float,
        z_nt: float,
        s_n: float,
        r_n: float,
        i_p: float,
        z_pl: float,
        z_pr: float,
        s_p: float,
        r_p: float,
        i_np: float,
        steel_grade: None | SteelGrade = None,
        **kwargs,
    ):
        super().__init__(section=section, current=current, steel_grade=steel_grade)

        self.designation = designation
        self.mass = mass
        self.section_shape = section_shape
        self.fabrication_type = fabrication_type

        self.b_1 = b_1
        self.b_2 = b_2
        self.t = t
        self.r_1 = r_1
        self.r_2 = r_2
        self.a_g = a_g
        self.n_l = n_l
        self.n_r = n_r
        self.p_b = p_b
        self.p_t = p_t
        self.i_x = i_x
        self.y_1 = y_1
        self.z_x1 = z_x1
        self.y_4 = y_4
        self.z_x4 = z_x4
        self.y_5 = y_5
        self.z_x5 = z_x5
        self.s_x = s_x
        self.r_x = r_x
        self.i_y = i_y
        self.x_5 = x_5
        self.z_y5 = z_y5
        self.x_3 = x_3
        self.z_y3 = z_y3
        self.x_2 = x_2
        self.z_y2 = z_y2
        self.s_y = s_y
        self.r_y = r_y
        self.j = j
        self.alpha = atan(tan_alpha)
        self.i_n = i_n
        self.z_nb = z_nb
        self.z_nt = z_nt
        self.s_n = s_n
        self.r_n = r_n
        self.i_p = i_p
        self.z_pl = z_pl
        self.z_pr = z_pr
        self.s_p = s_p
        self.r_p = r_p
        self.i_np = i_np

    @property
    def f_y(self):
        """
        Get the yield stress from the steel grade.
        """

        if self.grade is None:
            return None

        return self.grade.get_f_y(self.t)

    @property
    def f_u(self):
        """
        Get the ultimate stress from the steel grade.
        """

        if self.grade is None:
            return None

        return self.grade.get_f_u(self.t)


class RhsSectionData(SteelSection):
    """
    Class to map to an RHS / SHS section.
    """

    # TODO: fill out class.

    def __init__(
        self,
        *,
        section: str,
        current: bool,
        designation: str,
        mass: float,
        section_shape: float,
        fabrication_type: float,
        standard: str,
        b: float,
        d: float,
        t: float,
        r_ext: float,
        a_g: float,
        i_x: float,
        z_x: float,
        s_x: float,
        r_x: float,
        i_y: float,
        z_y: float,
        s_y: float,
        r_y: float,
        j: float,
        c: float,
        steel_grade: None | SteelGrade = None,
        **kwargs,
    ):
        super().__init__(section=section, current=current, steel_grade=steel_grade)

        self.designation = designation
        self.mass = mass
        self.section_shape = section_shape
        self.fabrication_type = fabrication_type
        self.standard = standard

        self.b = b
        self.d = d
        self.t = t
        self.r_ext = r_ext
        self.a_g = a_g
        self.i_x = i_x
        self.z_x = z_x
        self.s_x = s_x
        self.r_x = r_x
        self.i_y = i_y
        self.z_y = z_y
        self.s_y = s_y
        self.r_y = r_y
        self.j = j
        self.c = c

    @property
    def f_y(self):
        """
        Get the yield stress from the steel grade.
        """

        if self.grade is None:
            return None

        return self.grade.get_f_y(self.t)

    @property
    def f_u(self):
        """
        Get the ultimate stress from the steel grade.
        """

        if self.grade is None:
            return None

        return self.grade.get_f_u(self.t)


class ChsSectionData(SteelSection):
    """
    Class to map to a CHS section.
    """

    # TODO: fill out class.

    def __init__(
        self,
        *,
        section: str,
        current: bool,
        designation: str,
        mass: float,
        section_shape: float,
        fabrication_type: float,
        standard: str,
        d: float,
        t: float,
        a_g: float,
        i: float,
        z: float,
        s: float,
        r: float,
        j: float,
        c: float,
        steel_grade: None | SteelGrade = None,
        **kwargs,
    ):
        super().__init__(section=section, current=current, steel_grade=steel_grade)

        self.designation = designation
        self.mass = mass
        self.section_shape = section_shape
        self.fabrication_type = fabrication_type
        self.standard = standard

        self.d = d
        self.t = t
        self.a_g = a_g
        self.i = i
        self.z = z
        self.s = s
        self.r = r
        self.j = j
        self.c = c

    @property
    def f_y(self):
        """
        Get the yield stress from the steel grade.
        """

        if self.grade is None:
            return None

        return self.grade.get_f_y(self.t)

    @property
    def f_u(self):
        """
        Get the ultimate stress from the steel grade.
        """

        if self.grade is None:
            return None

        return self.grade.get_f_u(self.t)


def _grade_funcs(
    thickness_col: str,
    designation_col: str,
    steel_grade: SteelGrade | dict[str, SteelGrade] | None = None,
):
    """
    Helper function to create the functions that apply steel grades
    to steel section dataframes.

    :param section_df: The section dataframe to apply the grade properties to.
    :param steel_grade: An optional SteelGrade object or dictionary to assign
        to the sections. For different section types (e.g. WB vs UB),
        specify the grade as a dictionary: {designation: SteelGrade}.
        If a designation is missed, sections will be assigned a grade
        of None.
    """

    if isinstance(steel_grade, SteelGrade):
        # if provided a SteelGrade, choose f_y and f_u based on
        # thickness in the appropriate column.

        def fy_func(row):
            return steel_grade.get_f_y(row[thickness_col])

        def fu_func(row):
            return steel_grade.get_f_u(row[thickness_col])

    else:
        # if steel_grade is None, attempt to get the grade
        # from grade information in the dataframe.
        if steel_grade is None:
            steel_grade = steel_grades()

        def fy_func(row):
            designation = row[designation_col]

            if designation is None:
                return None

            grade_obj = steel_grade[designation]

            return grade_obj.get_f_y(row[thickness_col])

        def fu_func(row):
            designation = row[designation_col]

            if designation is None:
                return None

            grade_obj = steel_grade[designation]

            return grade_obj.get_f_u(row[thickness_col])

    return fy_func, fu_func


def i_section_df(
    steel_grade: None | SteelGrade | dict[str, SteelGrade] = None,
) -> pd.DataFrame:
    """
    Get a Pandas Dataframe with all the Australian Standard I sections.

    :param steel_grade: An optional SteelGrade object or dictionary to assign
        to the sections. For different section types (e.g. WB vs UB),
        specify the grade as a dictionary: {designation: SteelGrade}.
        If a designation is missed, sections will be assigned a grade
        of None.
    """

    section_df = pl.read_excel(_DATA_PATH / Path("steel_data.xlsx"), sheet_name="is")

    if steel_grade is None:
        return section_df.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("f_yf"),
                pl.lit(None).cast(pl.Float64).alias("f_yw"),
                pl.lit(None).cast(pl.Float64).alias("f_uf"),
                pl.lit(None).cast(pl.Float64).alias("f_uw"),
            ]
        )

    fyf, fuf = _grade_funcs(
        thickness_col="t_f", designation_col="designation", steel_grade=steel_grade
    )
    fyw, fuw = _grade_funcs(
        thickness_col="t_w", designation_col="designation", steel_grade=steel_grade
    )

    return section_df.with_columns(
        [
            pl.struct(["designation", "t_f"])
            .map_elements(fyf, return_dtype=pl.Float64)
            .alias("f_yf"),
            pl.struct(["designation", "t_w"])
            .map_elements(fyw, return_dtype=pl.Float64)
            .alias("f_yw"),
            pl.struct(["designation", "t_f"])
            .map_elements(fuf, return_dtype=pl.Float64)
            .alias("f_uf"),
            pl.struct(["designation", "t_w"])
            .map_elements(fuw, return_dtype=pl.Float64)
            .alias("f_uw"),
        ]
    )


def i_sections(
    steel_grade: None | SteelGrade | dict[str, SteelGrade] = None,
) -> dict[str, ISectionData]:
    """
    Build a dictionary of all the standard Australian I sections.

    :param steel_grade: An optional SteelGrade object or dictionary to assign
        to the sections. For different section types (e.g. WB vs UB),
        specify the grade as a dictionary: {designation: SteelGrade}.
        If a designation is missed, sections will be assigned a grade
        of None.
    """

    i_sects = {}

    i_sect_df = i_section_df()
    i_sect_df = i_sect_df.filter(pl.col("current") == "yes")

    for row in i_sect_df.to_dicts():
        sg = steel_grade

        if isinstance(sg, dict):
            sg = steel_grade.get(row["designation"])

        i_sects[row["section"]] = ISectionData(**row, steel_grade=sg)

    return i_sects


def c_section_df(
    steel_grade: None | SteelGrade | dict[str, SteelGrade] = None,
) -> pl.DataFrame:
    """
    Get a Polars DataFrame with all the Australian Standard C sections.

    :param steel_grade: An optional SteelGrade object or dictionary to assign
        to the sections. For different section types (e.g. WB vs UB),
        specify the grade as a dictionary: {designation: SteelGrade}.
        If a designation is missed, sections will be assigned a grade
        of None.
    """

    section_df = pl.read_excel(_DATA_PATH / Path("steel_data.xlsx"), sheet_name="cs")

    if steel_grade is None:
        return section_df.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("f_yf"),
                pl.lit(None).cast(pl.Float64).alias("f_yw"),
                pl.lit(None).cast(pl.Float64).alias("f_uf"),
                pl.lit(None).cast(pl.Float64).alias("f_uw"),
            ]
        )

    fyf, fuf = _grade_funcs(
        thickness_col="t_f", designation_col="designation", steel_grade=steel_grade
    )
    fyw, fuw = _grade_funcs(
        thickness_col="t_w", designation_col="designation", steel_grade=steel_grade
    )

    return section_df.with_columns(
        [
            pl.struct(["designation", "t_f"])
            .map_elements(fyf, return_dtype=pl.Float64)
            .alias("f_yf"),
            pl.struct(["designation", "t_w"])
            .map_elements(fyw, return_dtype=pl.Float64)
            .alias("f_yw"),
            pl.struct(["designation", "t_f"])
            .map_elements(fuf, return_dtype=pl.Float64)
            .alias("f_uf"),
            pl.struct(["designation", "t_w"])
            .map_elements(fuw, return_dtype=pl.Float64)
            .alias("f_uw"),
        ]
    )


def c_sections(
    steel_grade: None | SteelGrade | dict[str, SteelGrade] = None,
) -> dict[str, CSectionData]:
    """
    Build a dictionary of the standard Australian C sections.
    :param steel_grade: An optional SteelGrade object or dictionary to assign
        to the sections. For different section types (e.g. WB vs UB),
        specify the grade as a dictionary: {designation: SteelGrade}.
        If a designation is missed, sections will be assigned a grade
        of None.
        Note that currently Australian channels only come in one
        designation ("PFC") so typically a pure grade object would be passed in.
    """

    c_sects = {}

    for _, *v in c_section_df().iterrows():
        obj = v[0].to_dict()

        sg = steel_grade
        obj["current"] = obj["current"] == "yes"

        if isinstance(sg, dict):
            sg = steel_grade.get(obj["designation"])

        c_sects[obj["section"]] = CSectionData(**obj, steel_grade=sg)

    return c_sects


def angle_section_df(
    steel_grade: None | SteelGrade | dict[str, SteelGrade] = None,
) -> pl.DataFrame:
    """
    Get a Polars DataFrame with all the Australian Standard Angle sections.

    :param steel_grade: An optional SteelGrade object or dictionary to assign
        to the sections. For different section types (e.g. WB vs UB),
        specify the grade as a dictionary: {designation: SteelGrade}.
        If a designation is missed, sections will be assigned a grade
        of None.
    """

    section_df = pl.read_excel(
        _DATA_PATH / Path("steel_data.xlsx"), sheet_name="angles"
    )

    if steel_grade is None:
        return section_df.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("f_y"),
                pl.lit(None).cast(pl.Float64).alias("f_u"),
            ]
        )

    fy_func, fu_func = _grade_funcs(
        thickness_col="t", designation_col="designation", steel_grade=steel_grade
    )

    return section_df.with_columns(
        [
            pl.struct(["designation", "t"])
            .map_elements(fy_func, return_dtype=pl.Float64)
            .alias("f_y"),
            pl.struct(["designation", "t"])
            .map_elements(fu_func, return_dtype=pl.Float64)
            .alias("f_u"),
        ]
    )


def angle_sections(
    steel_grade: None | SteelGrade | dict[str, SteelGrade] = None,
) -> dict[str, AngleSectionData]:
    """
    Build a dictionary of the standard Australian angle sections.

    :param steel_grade: An optional SteelGrade object or dictionary to assign
        to the sections. For different section types (e.g. WB vs UB),
        specify the grade as a dictionary: {designation: SteelGrade}.
        If a designation is missed, sections will be assigned a grade
        of None.
        Note that currently Australian channels only come in one
        designation ("PFC") so typically a pure grade object would be passed in.
    """

    angle_sects = {}

    for _, *v in angle_section_df().iterrows():
        obj = v[0].to_dict()

        sg = steel_grade
        obj["current"] = obj["current"] == "yes"

        if isinstance(sg, dict):
            sg = steel_grade.get(obj["designation"])

        angle_sects[obj["section"]] = AngleSectionData(**obj, steel_grade=sg)

    return angle_sects


def rhs_section_df(
    steel_grade: None | SteelGrade | dict[str, SteelGrade] = None,
) -> pl.DataFrame:
    """
    Get a Polars DataFrame with all the Australian Standard RHS & SHS sections.

    :param steel_grade: An optional SteelGrade object or dictionary to assign
        to the sections. For different section types (e.g. WB vs UB),
        specify the grade as a dictionary: {designation: SteelGrade}.
        If a designation is missed, sections will be assigned a grade
        of None.
    """

    section_df = pl.read_excel(_DATA_PATH / Path("steel_data.xlsx"), sheet_name="rhs")

    if steel_grade is None:
        return section_df.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("f_y"),
                pl.lit(None).cast(pl.Float64).alias("f_u"),
            ]
        )

    fy_func, fu_func = _grade_funcs(
        thickness_col="t", designation_col="designation", steel_grade=steel_grade
    )

    return section_df.with_columns(
        [
            pl.struct(["designation", "t"])
            .map_elements(fy_func, return_dtype=pl.Float64)
            .alias("f_y"),
            pl.struct(["designation", "t"])
            .map_elements(fu_func, return_dtype=pl.Float64)
            .alias("f_u"),
        ]
    )


def rhs_sections(
    steel_grade: None | SteelGrade | dict[str, SteelGrade] = None,
) -> dict[str, RhsSectionData]:
    """
    Build a dictionary of the standard Australian angle sections.

    :param steel_grade: An optional SteelGrade object or dictionary to assign
        to the sections. For different section types (e.g. WB vs UB),
        specify the grade as a dictionary: {designation: SteelGrade}.
        If a designation is missed, sections will be assigned a grade
        of None.
        Note that currently Australian channels only come in one
        designation ("PFC") so typically a pure grade object would be passed in.
    """

    rhs_sects = {}

    for _, *v in rhs_section_df().iterrows():
        obj = v[0].to_dict()

        sg = steel_grade
        obj["current"] = obj["current"] == "yes"

        if isinstance(sg, dict):
            sg = steel_grade.get(obj["designation"])

        rhs_sects[obj["section"] + ":[" + obj["grade"] + "]"] = RhsSectionData(
            **obj, steel_grade=sg
        )

    return rhs_sects


def chs_section_df(
    steel_grade: None | SteelGrade | dict[str, SteelGrade] = None,
) -> pl.DataFrame:
    """
    Get a Polars DataFrame with all the Australian Standard CHS sections.

    :param steel_grade: An optional SteelGrade object or dictionary to assign
        to the sections. For different section types (e.g. WB vs UB),
        specify the grade as a dictionary: {designation: SteelGrade}.
        If a designation is missed, sections will be assigned a grade
        of None.
    """

    section_df = pl.read_excel(_DATA_PATH / Path("steel_data.xlsx"), sheet_name="chs")

    if steel_grade is None:
        return section_df.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("f_y"),
                pl.lit(None).cast(pl.Float64).alias("f_u"),
            ]
        )

    fy_func, fu_func = _grade_funcs(
        thickness_col="t", designation_col="designation", steel_grade=steel_grade
    )

    return section_df.with_columns(
        [
            pl.struct(["designation", "t"])
            .map_elements(fy_func, return_dtype=pl.Float64)
            .alias("f_y"),
            pl.struct(["designation", "t"])
            .map_elements(fu_func, return_dtype=pl.Float64)
            .alias("f_u"),
        ]
    )


def chs_sections(
    steel_grade: None | SteelGrade | dict[str, SteelGrade] = None,
) -> dict[str, ChsSectionData]:
    """
    Build a dictionary of the standard Australian CHS sections.

    :param steel_grade: An optional SteelGrade object or dictionary to assign
        to the sections. For different section types (e.g. WB vs UB),
        specify the grade as a dictionary: {designation: SteelGrade}.
        If a designation is missed, sections will be assigned a grade
        of None.
        Note that currently Australian channels only come in one
        designation ("PFC") so typically a pure grade object would be passed in.
    """

    chs_sects = {}

    for _, *v in chs_section_df().iterrows():
        obj = v[0].to_dict()

        sg = steel_grade
        obj["current"] = obj["current"] == "yes"

        if isinstance(sg, dict):
            sg = steel_grade.get(obj["designation"])

        chs_sects[obj["section"] + ":[" + obj["grade"] + "]"] = ChsSectionData(
            **obj, steel_grade=sg
        )

    return chs_sects


def standard_plate_df():
    """
    Load a dataframe of standard plate thicknesses.
    """

    return pd.read_excel(
        _DATA_PATH / Path("steel_data.xlsx"), sheet_name="standard_plate"
    )


def nearest_standard_plate(
    thickness,
    *,
    min_thickness: float | None = None,
    greater_than: bool = True,
    plate_df: pd.DataFrame | None = None,
):
    """
    Return the nearest standard plate to the specified thickness.

    :param thickness: The thickness to test.
    :param greater_than: If True, return the next largest or equal plate,
        If False return the next smallest or equal plate.
    :param plate_df: Optionally pass a Dataframe of plates in rather than using
        the one generated by standard_plate_df.
        The intent is to allow a DF with units attached to be passed in
        as well if required.
    """

    if min_thickness is not None:
        thickness = max(thickness, min_thickness)

    if plate_df is None:
        plate_df = standard_plate_df()

    if not greater_than:
        mask = plate_df.thickness.le(thickness)
        return plate_df.loc[mask, "thickness"].max()

    mask = plate_df.thickness.ge(thickness)
    return plate_df.loc[mask, "thickness"].min()


def standard_weld_df():
    """
    Get a DataFrame containing standard fillet weld sizes.
    """

    return pd.read_excel(
        _DATA_PATH / Path("steel_data.xlsx"), sheet_name="standard_fillet_welds"
    )


def nearest_standard_weld(
    size, *, greater_than: bool = True, weld_df: pd.DataFrame | None = None
):
    """
    Return the nearest standard weld size to the specified size.

    :param size: The size to test.
    :param greater_than: If True, return the next largest or equal weld,
        If False return the next smallest or equal weld.
    :param weld_df: Optionally pass a Dataframe of welds in rather than using
        the one generated by standard_weld_df.
        The intent is to allow a DF with units attached to be passed in
        as well if required.
    """

    if weld_df is None:
        weld_df = standard_weld_df()

    if not greater_than:
        mask = weld_df.leg_size.le(size)
        return weld_df.loc[mask, "leg_size"].max()

    mask = weld_df.leg_size.ge(size)
    return weld_df.loc[mask, "leg_size"].min()


def t_tp_chamfer(*, t_w, alpha, t_l=0, use_radians: bool = True):
    """
    Calculate the effective throat thickness of a chamfered plate.

    :param t_w: The size of the weld leg welded to the plate.
    :param alpha: The angle of the plate chamfer.
    :param t_l: The size of any landing on the chamfer.
    :param use_radians: Is alpha in radians or degrees.
    """

    if not use_radians:
        alpha = radians(alpha)

    if alpha < 0 or alpha > pi / 2:
        raise ValueError(
            f"alpha should be within the range of 0-90 degrees. alpha={degrees(alpha)}"
        )

    return t_w * sin(alpha) + t_l * cos(alpha)


def jack_bolt_effort(*, load, p, r_effort, r_bolt, mu, against: bool = True):
    """
    Calculate the force required to turn a jacking bolt.

    :param load: The weight or load.
    :param p: The pitch of the screw
    :param r_effort: The radius of the effort force.
    :param r_bolt: The radius of the screw / bolt.
    :param mu: The assumed coefficient of friction.
        Note that for bolts this can vary dramatically depending on thread condition.
    :param against: Is the jacking against the load or with the load?
        Jacking against the load (e.g. raising a load under gravity)
        requires larger force than jacking with the load (e.g. lowering a load under gravity).
    """

    if against:
        return (
            load
            * ((2 * pi * mu * r_bolt + p) / (2 * pi * r_bolt - mu * p))
            * (r_bolt / r_effort)
        )

    return (
        load
        * ((2 * pi * mu * r_bolt - p) / (2 * pi * r_bolt + mu * p))
        * (r_bolt / r_effort)
    )


def local_thickness_reqd(
    *, point_force, f_y, phi=0.9, width_lever: tuple[float, float] | None = None
):
    """
    Calculate the local thickness of a plate required to transfer a load back to a support by local bending.
    This assumes a 45deg load dispersion through the plate towards the support, ignoring and load transfer into
    the 2nd dimension.

    :param point_force: The load to transfer.
    :param f_y: The yield stress of the plate.
    :param phi: A capacity reduction factor.
    :param width_lever: A tuple containing the width of the applied load and the lever arm between load and support.
        If None, the load is treated as a point load.
    """

    if width_lever is None:
        return ((2 * point_force) / (phi * f_y)) ** 0.5

    lw = width_lever[0]
    lever = width_lever[1]

    width_lever_ratio = 0 if lever == 0 else lw / lever

    return 2 / (((phi * f_y / point_force) * (width_lever_ratio + 2)) ** 0.5)


def make_section(
    *, geometry: Geometry, alpha_mesh: float = 100, calculate_properties: bool = True
) -> Section:
    """
    Turn a sectionproperties Geometry object into a Section object.

    :param geometry: The Geometry object to turn into the Section object.
    :param alpha_mesh: The area of the largest mesh element,
        as a fraction of the overall area of the section.
    :param calculate_properties: Calculate the properties?
    """

    area = geometry.calculate_area()
    sect = Section(geometry.create_mesh(mesh_sizes=area / alpha_mesh))

    if calculate_properties:
        sect.calculate_geometric_properties()
        sect.calculate_warping_properties()
        sect.calculate_plastic_properties()

    return sect


def make_i_section(
    *,
    b_f,
    d,
    t_f,
    t_w,
    b_fb=None,
    t_fb=None,
    corner_radius=None,
    weld_size=None,
    n_r=8,
    alpha_mesh=100,
    calculate_properties=True,
) -> Section:
    """
    Generate an I Section using the section-properties library.
    A helper wrapper around the exist. i section functions.

    :param b_f: The width of the top flange.
    :param d: The section depth.
    :param t_f: The top flange thickness.
    :param t_w: The web thickness.
    :param b_fb: The width of the bottom section.
        Provide None for a symmetric section.
    :param t_fb: The bottom flange thickness.
        Provide None if both flanges the same.
    :param corner_radius: The corner radius of the i-section.
        NOTE: If both a corner radius and a weld size are defined,
        priority is given to corner radii.
        If a weld is present, set corner_radius = None.
    :param weld_size: The corner fillet weld (if any).
        NOTE: If both a corner radius and a weld size are defined,
        priority is given to corner radii.
        If a weld is present, set corner_radius = None.
    :param n_r: The no. of points used to model the corner.
    :param alpha_mesh: The area of the largest mesh element,
        as a fraction of the overall area of the section.
    :param calculate_properties: Calculate the properties?
    """

    geometry = make_i_geometry(
        b_f=b_f,
        d=d,
        t_f=t_f,
        t_w=t_w,
        b_fb=b_fb,
        t_fb=t_fb,
        corner_radius=corner_radius,
        weld_size=weld_size,
        n_r=n_r,
    )

    return make_section(
        geometry=geometry,
        alpha_mesh=alpha_mesh,
        calculate_properties=calculate_properties,
    )


def make_i_geometry(
    *, b_f, d, t_f, t_w, b_fb=None, t_fb=None, corner_radius=None, weld_size=None, n_r=8
):
    """
    Generate an I Section using the section-properties library.
    A helper wrapper around the exist. i section functions.

    :param b_f: The width of the top flange.
    :param d: The section depth.
    :param t_f: The top flange thickness.
    :param t_w: The web thickness.
    :param b_fb: The width of the bottom section.
        Provide None for a symmetric section.
    :param t_fb: The bottom flange thickness.
        Provide None if both flanges the same.
    :param corner_radius: The corner radius of the i-section.
        NOTE: If both a corner radius and a weld size are defined,
        priority is given to corner radii.
        If a weld is present, set corner_radius = None.
    :param weld_size: The corner fillet weld (if any).
        NOTE: If both a corner radius and a weld size are defined,
        priority is given to corner radii.
        If a weld is present, set corner_radius = None.
    """

    b_ft = b_f
    t_ft = t_f

    if b_fb is None:
        b_fb = b_f
    if t_fb is None:
        t_fb = t_f

    if corner_radius is None:
        corner_radius = 0

        weld = triangular_section(b=weld_size, h=weld_size)
        bottom_right = weld
        bottom_left = weld.mirror_section(axis="y").shift_section(
            x_offset=-weld_size, y_offset=0
        )
        top_right = bottom_right.mirror_section().shift_section(
            x_offset=0, y_offset=-weld_size
        )
        top_left = bottom_left.mirror_section().shift_section(
            x_offset=0, y_offset=-weld_size
        )

    geometry = mono_i_section(
        d=d,
        b_t=b_ft,
        b_b=b_fb,
        t_ft=t_ft,
        t_fb=t_fb,
        t_w=t_w,
        r=corner_radius,
        n_r=n_r,
    )

    if weld_size is None:
        return geometry

    xc, yc = geometry.calculate_centroid()

    bottom_right = bottom_right.shift_section(x_offset=t_w / 2, y_offset=t_fb)
    bottom_left = bottom_left.shift_section(x_offset=-t_w / 2, y_offset=t_fb)
    top_right = top_right.shift_section(x_offset=t_w / 2, y_offset=d - t_ft)
    top_left = top_left.shift_section(x_offset=-t_w / 2, y_offset=d - t_ft)

    geometry = geometry.shift_section(x_offset=-xc, y_offset=0)
    geometry = (geometry - bottom_right) + bottom_right
    geometry = (geometry - bottom_left) + bottom_left
    geometry = (geometry - top_right) + top_right
    return (geometry - top_left) + top_left


def eff_length(
    *,
    elastic_modulus,
    second_moment,
    buckling_load,
    k_e: bool = False,
    actual_length=None,
):
    """
    Calculate the effective length back-calculated from the buckling load.

    :param elastic_modulus: The elastic modulus of the section.
    :param second_moment: The second moment of area about the axis which is buckling.
    :param buckling_load: The buckling load.
    :param k_e: Calculate the length factor k_e rather than the buckling length.
    :param actual_length: The actual length of the section. Only used if k_e=True.
    """

    kel = (((pi**2) * elastic_modulus * second_moment) / buckling_load) ** 0.5

    if k_e:
        if actual_length is None:
            raise ValueError(
                f"To calculate k_e the actual length is required. {actual_length=}"
            )

        return kel / actual_length

    return kel


class OverplatedSection:
    """
    Creates an overplated section with an overplate on the top & bottom.

    NOTE: Overplates are centred on the section.
    NOTE 2: This will work best with a symmetric or mono-symmetric I section,
        with other sections your mileage may vary until I update it.
    """

    def __init__(
        self,
        base_section,
        t_op_top=None,
        b_op_top=None,
        t_op_bottom=None,
        b_op_bottom=None,
        alpha_mesh_size=200,
    ):
        """
        Create an overplated I section.

        :param base_section: The section to overplate.
        :param t_op_top: The thickness of the top overplate.
            Use None if no overplate on top.
        :param b_op_top: The width of the top overplate.
            Use None if no overplate on top.
        :param t_op_bottom: The thickness of the bottom overplate.
            Use None if no overplate on bottom.
        :param b_op_bottom: The width of the bottom overplate.
            Use None if no overplate on the bottom.
        :param alpha_mesh_size: How big should the largest mesh element be,
            as a fraction of the total area of the shape?
        """

        self.base_section = base_section.align_center(align_to=(0, 0))
        self.t_op_top = t_op_top
        self.b_op_top = b_op_top
        self.t_op_bottom = t_op_bottom
        self.b_op_bottom = b_op_bottom
        self.alpha_mesh_size = alpha_mesh_size

        self._combined_geometry = None
        self._combined_sect = None

    @property
    def base_depth(self) -> float:
        """
        The depth of the base section.
        """

        extents = self.base_section.calculate_extents()

        return extents[3] - extents[2]

    @property
    def op_top(self) -> Geometry:
        if self.t_op_top is None:
            return None

        if self.b_op_top is None:
            return None

        return rectangular_section(d=self.t_op_top, b=self.b_op_top).align_center(
            align_to=(0, 0)
        )

    @property
    def op_bottom(self) -> Geometry:
        if self.t_op_bottom is None:
            return None
        if self.b_op_bottom is None:
            return None

        return rectangular_section(d=self.t_op_bottom, b=self.b_op_bottom).align_center(
            align_to=(0, 0)
        )

    @property
    def combined_geometry(self) -> Geometry:
        if self._combined_geometry is not None:
            return self._combined_geometry

        extents = self.base_section.calculate_extents()

        if self.op_top is None and self.op_bottom is None:
            comb_sect = self.base_section

        elif self.op_top is None:
            comb_sect = self.base_section + self.op_bottom.shift_section(
                y_offset=extents[2] - self.t_op_bottom / 2
            )
        elif self.op_bottom is None:
            comb_sect = self.base_section + self.op_top.shift_section(
                y_offset=extents[3] + self.t_op_top / 2
            )
        else:
            comb_sect = (
                self.base_section
                + self.op_top.shift_section(y_offset=extents[3] + self.t_op_top / 2)
                + self.op_bottom.shift_section(
                    y_offset=extents[2] - self.t_op_bottom / 2
                )
            )

        self._combined_geometry = comb_sect.align_to(other=(0, 0), on="top")

        return self._combined_geometry

    @property
    def combined_area(self) -> float:
        return self.combined_geometry.calculate_area()

    @property
    def combined_sect(self) -> Section:
        if self._combined_sect is not None:
            return self._combined_sect

        mesh_size = self.combined_area / self.alpha_mesh_size
        sect = Section(self.combined_geometry.create_mesh(mesh_sizes=mesh_size))

        sect.calculate_geometric_properties()
        sect.calculate_plastic_properties()
        sect.calculate_warping_properties()

        self._combined_sect = sect

        return self._combined_sect

    @property
    def d_total(self) -> float:
        if self.t_op_top is None and self.t_op_bottom is None:
            return self.base_depth

        if self.t_op_top is None:
            return self.base_depth + self.t_op_bottom

        if self.t_op_bottom is None:
            return self.base_depth + self.t_op_top

        return self.base_depth + self.t_op_top + self.t_op_bottom

    @property
    def yc_op_top(self) -> float:
        if self.op_top is None:
            return None

        return self.d_total - self.t_op_top / 2

    @property
    def yc_op_bottom(self) -> float:
        if self.t_op_bottom is None:
            return None

        return self.t_op_bottom / 2

    @property
    def q_op_top(self) -> float:
        if self.t_op_top is None:
            return None

        lever_arm = self.yc_op_top - self.combined_geometry.calculate_centroid()[1]
        area = self.op_top.calculate_area()
        return area * lever_arm

    @property
    def q_op_bottom(self) -> float:
        if self.t_op_bottom is None:
            return None

        lever_arm = self.combined_geometry.calculate_centroid()[1] - self.yc_op_bottom
        area = self.op_bottom.calculate_area()
        return area * lever_arm

    def plot_geometry(self):
        """
        Plot the geometry of the overplated section.

        Simply passes through to the plot_geometry() method of the combined_geometry.
        """

        self.combined_geometry.plot_geometry()


class BoltGroup:
    def __init__(self, bolts: list[tuple[float, float]]):
        x_bolts = np.asarray([b[0] for b in bolts])
        y_bolts = np.asarray([b[1] for b in bolts])

        x_c = np.average(x_bolts, axis=0)
        y_c = np.average(y_bolts, axis=0)

        x_bolts = x_bolts - x_c
        y_bolts = y_bolts - y_c

        self._x_bolts = x_bolts
        self._y_bolts = y_bolts
        self._x_offset = x_c
        self._y_offset = y_c

    @property
    def n_bolts(self):
        return len(self.bolts)

    @property
    def bolts(self):
        """
        Return a list of the bolts that make up the section,
        in the original co-ordinate system.
        """

        return list(zip(self.x_bolts, self.y_bolts))

    @property
    def bolts_c(self):
        """
        Return a list of bolts that make up the section, about the centroid.
        """

        return list(zip(self.x_bolts_c, self.y_bolts_c))

    @property
    def x_bolts(self):
        """
        Return the x-coordinates of the bolts in the original system.
        """

        return self._x_bolts + self._x_offset

    @property
    def y_bolts(self):
        """
        Return the y-coordinates of the bolts in the original system.
        """

        return self._y_bolts + self._y_offset

    @property
    def x_bolts_c(self):
        """
        Return the x-coordinates of the bolts about the centroid.
        """

        return self._x_bolts

    @property
    def y_bolts_c(self):
        """
        Return the y-coordinates of the bolts about the centroid.
        """

        return self._y_bolts

    @property
    def r_bolts_c(self):
        """
        Return the radius of the bolts about the group centroid.
        """

        return (self.x_bolts_c**2 + self.y_bolts_c**2) ** 0.5

    @property
    def r_bolts(self):
        """
        Return the radius of the bolts about the x-y origin.
        """

        return (self.x_bolts**2 + self.y_bolts**2) ** 0.5

    @property
    def i_xx(self):
        """
        Return the moment of inertia about the x-x axis.
        """

        return np.sum(self.y_bolts**2)

    @property
    def i_yy(self):
        """
        Return the moment of inertia about the y-y axis.
        """

        return np.sum(self.x_bolts**2)

    @property
    def i_zz(self):
        """
        Return the polar moment of inertia of the bolt group, about the x-y origin.
        """

        return np.sum(self.r_bolts**2)

    @property
    def i_uu_c(self):
        """
        Return the moment of inertia about the u-u axis through the bolt group centroid.
        """

        return np.sum(self.y_bolts_c**2)

    @property
    def i_vv_c(self):
        """
        Return the moment of inertia about the v-v axis through the bolt group centroid.
        """

        return np.sum(self.x_bolts_c**2)

    @property
    def i_ww_c(self):
        """
        Return the polar moment of inertia of the bolt group, about the centroid.
        """

        return np.sum(self.r_bolts_c**2)


def make_circular_bolt_group(centroid, radius, no_bolts):
    """
    Build a BoltGroup made out of a circular bolt pattern.

    :param centroid:
    :param radius:
    :param no_bolts:
    """

    return BoltGroup(
        bolts=build_circle(centroid=centroid, radius=radius, no_points=no_bolts + 1)[
            :-1
        ]
    )


def flat_plate_bending_uniform(*, a: float, b: float, t: float, e: float, q: float):
    """
    Calculate the maximum stress in a flat plate under uniform bending.

    Notes
    -----
    * This is based on Roark's stress & strain [1]_.
    * This assumes the plate is simply supported on all sides.

    Parameters
    ----------
    a : float
        The length (long side) of the plate.
    b : float
        The width (short side) of the plate.
    t : float
        The thickness of the plate.
    e : float
        The modulus of elasticity of the plate.
    q : float
        The uniform load on the plate.

    References
    ----------
    .. [1] Roark's Stress & Strain, 8th Ed, Table 11.4
    """

    a_b_ratio = np.asarray(
        [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0, 1000]
    )  # use 1000 for infinity
    beta_data = np.asarray(
        [0.2874, 0.3762, 0.4530, 0.5172, 0.5688, 0.6102, 0.7134, 0.7410, 0.7476, 0.7500]
    )
    alpha_data = np.asarray(
        [0.0444, 0.0616, 0.0770, 0.0906, 0.1017, 0.1110, 0.1335, 0.1400, 0.1417, 0.1421]
    )
    gamma_data = np.asarray(
        [0.420, 0.455, 0.478, 0.491, 0.499, 0.503, 0.505, 0.502, 0.501, 0.500]
    )

    if a < b:  # a_b_ratio needs to be > 1.0
        a, b = b, a

    alpha = np.interp(a / b, a_b_ratio, alpha_data)
    beta = np.interp(a / b, a_b_ratio, beta_data)
    gamma = np.interp(a / b, a_b_ratio, gamma_data)

    sigma_max = beta * q * b**2 / t**2
    y_max = alpha * q * b**4 / (e * t**3)
    r_max = gamma * q * b

    return {"sigma_max": sigma_max, "y_max": y_max, "R_max": r_max}


def flat_plate_bending_point(
    *, a: float, b: float, t: float, e: float, w: float, r_o: float
):
    """
    Calculate the maximum stress in a flat plate under a point load.

    Notes
    -----
    * This is based on Roark's stress & strain [1]_.
    * This assumes the plate is simply supported on all sides.
    * Assumes that poisson's ratio for steel is 0.3.

    Parameters
    ----------
    a : float
        The length (long side) of the plate.
    b : float
        The width (short side) of the plate.
    t : float
        The thickness of the plate.
    e : float
        The modulus of elasticity of the plate.
    w : float
        The point load on the plate.
    r_o : float
        The radius of the point load.

    References
    ----------
    .. [1] Roark's Stress & Strain, 8th Ed, Table 11.4
    """

    a_b_ratio = np.asarray([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 1000])
    beta_data = np.asarray([0.435, 0.650, 0.789, 0.875, 0.927, 0.958, 1.000])
    alpha_data = np.asarray([0.1267, 0.1478, 0.1621, 0.1715, 0.1770, 0.1805, 0.1851])

    if a < b:
        a, b = b, a

    alpha = np.interp(a / b, a_b_ratio, alpha_data)
    beta = np.interp(a / b, a_b_ratio, beta_data)

    r_prime_o = (1.6 * r_o**2 + t**2) ** 0.5 - 0.675 * t if r_o < 0.5 * t else r_o

    print(r_prime_o)

    sigma_max = ((3 * w) / (2 * pi * t**2)) * (
        (1.3) * log((2 * b) / (pi * r_prime_o)) + beta
    )
    y_max = alpha * w * b**2 / (e * t**3)

    return {"sigma_max": sigma_max, "y_max": y_max}
