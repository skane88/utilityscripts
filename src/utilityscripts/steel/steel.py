"""
To contain some utilities for steel design
"""

from __future__ import annotations

from math import cos, degrees, log, pi, radians, sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sectionproperties.analysis.section import Section
from sectionproperties.pre.geometry import Geometry
from sectionproperties.pre.library.primitive_sections import (
    rectangular_section,
    triangular_section,
)
from sectionproperties.pre.library.steel_sections import mono_i_section

from utilityscripts.geometry import build_circle

# All the standard I & C sections as dataframes
_DATA_PATH = Path(Path(__file__).parent) / Path("data")


def steel_grade_df() -> pl.DataFrame:
    """
    Get a Polars Dataframe with all the Australian Standard steel grades.
    """

    return pl.read_excel(_DATA_PATH / Path("steel_data.xlsx"), sheet_name="grades")


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
        Represents a steel material, with different yield strengths
        (and potentially ultimate strengths) at different thicknesses.

        Parameters
        ----------
        standard : str
            The standard the steel is made to.
        current : bool
            Is the steel a currently produced steel?
        form : str
            The form of the steel (e.g. plate, section etc.).
        grade : str
            The grade designation.
        thickness : array_like
            An array of thicknesses. Must be sorted smallest to largest.
            Typ. should start at 0 thick, or whatever minimum thickness
            the steel is produced in.
        f_y : array_like
            An array of yield strengths. Should be sorted to match thickness.
        f_u : array_like
            An array of ultimate strengths. Should be sorted to match thickness.
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

    def get_f_y(self, thickness: float) -> float:
        """
        Get the yield strength at a given thickness.

        Parameters
        ----------
        thickness : float
            The thickness to test.
        """

        return float(np.interp(x=thickness, xp=self.thickness, fp=self.f_y))

    def get_f_u(self, thickness: float) -> float:
        """
        Get the ultimate strength at a given thickness.

        Parameters
        ----------
        thickness : float
            The thickness to test.
        """

        return float(np.interp(x=thickness, xp=self.thickness, fp=self.f_u))

    def plot_grade(self, *, strength: str = "both"):
        """
        Plot the strength of the steel vs thickness.

        Parameters
        ----------
        strength : str
            Either 'f_y' or 'f_u' or 'both'
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

    sg_df = steel_grade_df().sort(["standard", "grade", "t"])

    # get a list of unique grades
    unique_grades = (
        sg_df[["standard", "current", "form", "grade"]]
        .unique(subset=["standard", "grade"])
        .sort(["standard", "grade"])
    )

    grades = {}

    for standard, current, form, grade in unique_grades.iter_rows():
        t = (
            sg_df.filter((pl.col("standard") == standard) & (pl.col("grade") == grade))
            .select("t")
            .to_numpy()
            .T[0]
        )
        f_y = (
            sg_df.filter((pl.col("standard") == standard) & (pl.col("grade") == grade))
            .select("f_y")
            .to_numpy()
            .T[0]
        )
        f_u = (
            sg_df.filter((pl.col("standard") == standard) & (pl.col("grade") == grade))
            .select("f_u")
            .to_numpy()
            .T[0]
        )

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
) -> pl.DataFrame:
    """
    Get a Polars Dataframe with all the Australian Standard I sections.

    Parameters
    ----------
    steel_grade : None | SteelGrade | dict[str, SteelGrade]
        An optional SteelGrade object or dictionary to assign
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


def c_section_df(
    steel_grade: None | SteelGrade | dict[str, SteelGrade] = None,
) -> pl.DataFrame:
    """
    Get a Polars DataFrame with all the Australian Standard C sections.

    Parameters
    ----------
    steel_grade : None | SteelGrade | dict[str, SteelGrade]
        An optional SteelGrade object or dictionary to assign
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


def angle_section_df(
    steel_grade: None | SteelGrade | dict[str, SteelGrade] = None,
) -> pl.DataFrame:
    """
    Get a Polars DataFrame with all the Australian Standard Angle sections.

    Parameters
    ----------
    steel_grade : None | SteelGrade | dict[str, SteelGrade]
        An optional SteelGrade object or dictionary to assign
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


def rhs_section_df(
    steel_grade: None | SteelGrade | dict[str, SteelGrade] = None,
) -> pl.DataFrame:
    """
    Get a Polars DataFrame with all the Australian Standard RHS & SHS sections.

    Parameters
    ----------
    steel_grade : None | SteelGrade | dict[str, SteelGrade]
        An optional SteelGrade object or dictionary to assign
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


def chs_section_df(
    steel_grade: None | SteelGrade | dict[str, SteelGrade] = None,
) -> pl.DataFrame:
    """
    Get a Polars DataFrame with all the Australian Standard CHS sections.

    Parameters
    ----------
    steel_grade : None | SteelGrade | dict[str, SteelGrade]
        An optional SteelGrade object or dictionary to assign
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


def standard_plate_df():
    """
    Load a dataframe of standard plate thicknesses.
    """

    return pl.read_excel(
        _DATA_PATH / Path("steel_data.xlsx"), sheet_name="standard_plate"
    )


def nearest_standard_plate(
    thickness,
    *,
    min_thickness: float | None = None,
    greater_than: bool = True,
    plate_df: pl.DataFrame | None = None,
):
    """
    Return the nearest standard plate to the specified thickness.

    Parameters
    ----------
    thickness : float
        The thickness to test.
    greater_than : bool
        If True, return the next largest or equal plate,
        If False return the next smallest or equal plate.
    plate_df : pd.DataFrame | None
        Optionally pass a Dataframe of plates in rather than using
        the one generated by standard_plate_df.
        The intent is to allow a DF with units attached to be passed in
        as well if required.
    """

    if min_thickness is not None:
        thickness = max(thickness, min_thickness)

    if plate_df is None:
        plate_df = standard_plate_df()

    if not greater_than:
        return (
            plate_df.filter(pl.col("thickness").le(thickness))
            .select("thickness")
            .max()
            .item()
        )

    return (
        plate_df.filter(pl.col("thickness").ge(thickness))
        .select("thickness")
        .min()
        .item()
    )


def standard_weld_df():
    """
    Get a DataFrame containing standard fillet weld sizes.
    """

    return pl.read_excel(
        _DATA_PATH / Path("steel_data.xlsx"), sheet_name="standard_fillet_welds"
    )


def nearest_standard_weld(
    size, *, greater_than: bool = True, weld_df: pl.DataFrame | None = None
):
    """
    Return the nearest standard weld size to the specified size.

    Parameters
    ----------
    size : float
        The size to test.
    greater_than : bool
        If True, return the next largest or equal weld,
        If False return the next smallest or equal weld.
    weld_df : pd.DataFrame | None
        Optionally pass a Dataframe of welds in rather than using
        the one generated by standard_weld_df.
        The intent is to allow a DF with units attached to be passed in
        as well if required.
    """

    if weld_df is None:
        weld_df = standard_weld_df()

    if not greater_than:
        return (
            weld_df.filter(pl.col("leg_size").le(size)).select("leg_size").max().item()
        )

    return weld_df.filter(pl.col("leg_size").ge(size)).select("leg_size").min().item()


def t_tp_chamfer(*, t_w, alpha, t_l=0, use_radians: bool = True):
    """
    Calculate the effective throat thickness of a chamfered plate.

    Parameters
    ----------
    t_w : float
        The size of the weld leg welded to the plate.
    alpha : float
        The angle of the plate chamfer.
    t_l : float
        The size of any landing on the chamfer.
    use_radians : bool
        Is alpha in radians or degrees.
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

    Parameters
    ----------
    load : float
        The weight or load.
    p : float
        The pitch of the screw
    r_effort : float
        The radius of the effort force.
    r_bolt : float
        The radius of the screw / bolt.
    mu : float
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

    Parameters
    ----------
    point_force : float
        The load to transfer.
    f_y : float
        The yield stress of the plate.
    phi : float
        A capacity reduction factor.
    width_lever : tuple[float, float] | None
        A tuple containing the width of the applied load and the lever arm between load and support.
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

    Parameters
    ----------
    geometry : Geometry
        The Geometry object to turn into the Section object.
    alpha_mesh : float
        The area of the largest mesh element,
        as a fraction of the overall area of the section.
    calculate_properties : bool
        Calculate the properties?
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

    Parameters
    ----------
    b_f : float
        The width of the top flange.
    d : float
        The section depth.
    t_f : float
        The top flange thickness.
    t_w : float
        The web thickness.
    b_fb : float | None
        The width of the bottom section.
        Provide None for a symmetric section.
    :param t_fb: The bottom flange thickness.
        Provide None if both flanges the same.
    corner_radius : float | None
        The corner radius of the i-section.
        NOTE: If both a corner radius and a weld size are defined,
        priority is given to corner radii.
        If a weld is present, set corner_radius = None.
    weld_size : float | None
        The corner fillet weld (if any).
        NOTE: If both a corner radius and a weld size are defined,
        priority is given to corner radii.
        If a weld is present, set corner_radius = None.
    n_r : int
        The no. of points used to model the corner.
    alpha_mesh : float
        The area of the largest mesh element,
        as a fraction of the overall area of the section.
    calculate_properties : bool
        Calculate the properties?
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

    Parameters
    ----------
    b_f : float
        The width of the top flange.
    d : float
        The section depth.
    t_f : float
        The top flange thickness.
    t_w : float
        The web thickness.
    b_fb : float | None
        The width of the bottom section.
        Provide None for a symmetric section.
    t_fb : float | None
        The bottom flange thickness.
        Provide None if both flanges the same.
    corner_radius : float | None
        The corner radius of the i-section.
        NOTE: If both a corner radius and a weld size are defined,
        priority is given to corner radii.
        If a weld is present, set corner_radius = None.
    weld_size : float | None
        The corner fillet weld (if any).
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

    xc, _yc = geometry.calculate_centroid()

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

    Parameters
    ----------
    elastic_modulus : float
        The elastic modulus of the section.
    second_moment : float
        The second moment of area about the axis which is buckling.
    buckling_load : float
        The buckling load.
    k_e : bool
        Calculate the length factor k_e rather than the buckling length.
    actual_length : float | None
        The actual length of the section. Only used if k_e=True.
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

    Notes
    -----
    - Overplates are centred on the section.
    - This will work best with a symmetric or mono-symmetric I section,
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

        Parameters
        ----------
        base_section : Section
            The section to overplate.
        t_op_top : float | None
            The thickness of the top overplate.
            Use None if no overplate on top.
        b_op_top : float | None
            The width of the top overplate.
            Use None if no overplate on top.
        t_op_bottom : float | None
            The thickness of the bottom overplate.
            Use None if no overplate on bottom.
        b_op_bottom : float | None
            The width of the bottom overplate.
            Use None if no overplate on the bottom.
        alpha_mesh_size : float
            How big should the largest mesh element be,
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

        return list(zip(self.x_bolts, self.y_bolts, strict=True))

    @property
    def bolts_c(self):
        """
        Return a list of bolts that make up the section, about the centroid.
        """

        return list(zip(self.x_bolts_c, self.y_bolts_c, strict=True))

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


def bolt_grade_df():
    """
    Get a dataframe of the bolt grades.
    """

    return pl.read_excel(_DATA_PATH / Path("steel_data.xlsx"), sheet_name="bolt_grades")


class BoltGrade:
    """
    A class to represent a bolt grade.
    """

    def __init__(
        self,
        *,
        grade: str,
        diameters: list[float] | np.ndarray,
        f_yf: list[float] | np.ndarray,
        f_uf: list[float] | np.ndarray,
    ):
        """
        Initialise the BoltGrade class.

        Parameters
        ----------
        grade : str
            The grade designation.
        diameters : list[float] | np.ndarray
            The limiting diameters for the grade, in m.
        f_yf : list[float] | np.ndarray
            The yield strength of the bolts, in Pa. The length should match the
            length of the diameters array.
        f_uf : list[float] | np.ndarray
            The ultimate strength of the bolts, in Pa. The length should match the
            length of the diameters array.
        """

        self._grade = grade
        self._diameters = np.asarray(diameters)
        self._f_yf = np.asarray(f_yf)
        self._f_uf = np.asarray(f_uf)

    @property
    def grade(self) -> str:
        """
        The grade designation.
        """

        return self._grade

    @property
    def diameters(self) -> np.ndarray:
        """
        The limiting diameters for the grade, in m.
        """

        return self._diameters

    @property
    def f_yf(self) -> np.ndarray:
        """
        The yield strength of the bolts, in Pa.
        """

        return self._f_yf

    @property
    def f_uf(self) -> np.ndarray:
        """
        The ultimate strength of the bolts, in Pa.
        """

        return self._f_uf

    def get_f_yf(self, diameter: float) -> float:
        """
        Get the yield strength of the bolts, in Pa.

        Parameters
        ----------
        diameter : float
            The diameter of the bolt, in m.

        Returns
        -------
        float
            The yield strength of the bolts, in Pa.
        """

        return np.interp(diameter, self._diameters, self._f_yf)

    def get_f_uf(self, diameter: float) -> float:
        """
        Get the ultimate strength of the bolts, in Pa.

        Parameters
        ----------
        diameter : float
            The diameter of the bolt, in m.

        Returns
        -------
        float
            The ultimate strength of the bolts, in Pa.
        """

        return np.interp(diameter, self._diameters, self._f_uf)

    def __repr__(self):
        return f"{type(self).__name__} {self.grade}"


def bolt_grades() -> dict[str, BoltGrade]:
    """
    Get a dictionary of the standard bolt grades.

    Returns
    -------
    dict[str, BoltGrade]
        A dictionary of the standard bolt grades, with the designation as the key.
    """

    bolt_grade_data = bolt_grade_df().sort(["grade", "d_f"])

    # get a list of unique grades
    unique_grades = bolt_grade_data[["grade"]].unique(subset=["grade"]).sort(["grade"])

    grades = {}

    for grade in unique_grades.iter_rows():
        bg = bolt_grade_data.filter((pl.col("grade") == grade[0])).select(
            "d_f", "f_yf", "f_uf"
        )

        diameters = bg.select("d_f").to_numpy().T[0]
        f_yf = bg.select("f_yf").to_numpy().T[0]
        f_uf = bg.select("f_uf").to_numpy().T[0]

        grades[str(grade[0])] = BoltGrade(
            grade=str(grade[0]),
            diameters=diameters,
            f_yf=f_yf,
            f_uf=f_uf,
        )

    return grades
