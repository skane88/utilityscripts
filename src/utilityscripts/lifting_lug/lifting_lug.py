"""
Contains classes etc. to model a lifting lug.
"""

from math import asin, atan, cos, radians, sin

import matplotlib.pyplot as plt
import numpy as np
from shapely import LineString, Point, Polygon
from shapely.affinity import rotate
from shapely.ops import split
from shapely.plotting import plot_polygon

from utilityscripts.steel.steel import SteelGrade


class LugLoad:
    """
    A class to hold a definition for a load to apply to the Lug class.
    """

    def __init__(
        self,
        *,
        swl: float,
        out_of_plane_allowance: float,
        dia_pin: float,
        shackle_offset: float = 0.0,
        out_of_plane_offset: float = 0.0,
        in_plane_angle_limits: tuple[float, float] = (0.0, 0.0),
        out_of_plane_angle_limits: tuple[float, float] = (0.0, 0.0),
        use_radians: bool = True,
    ):
        """

        Parameters
        ----------
        swl : float
            The safe working load of the lug.
        out_of_plane_allowance : float
            An allowance for any unintended out-of-plane
            loading. Typ. value in AS1418 is 4% of the SWL.
            This is applied in addition to the calculated out-of-plane load.
        dia_pin : float
            The diameter of the pin used.
        shackle_offset : float
            Any offset from the hole centre in the direction towards
            the load. For example due to a shackle etc.
        out_of_plane_offset : float
            Any horizontal offset out-of-plane.
            For example, if using shackles too wide for the lug.
        in_plane_angle_limits : tuple[float, float]
            The minimum and maximum angle of the load, in plane.
            A tuple of the form (min, max). +ve is CCW.
        out_of_plane_angle_limits : tuple[float, float]
            The maximum angle of the load, in plane.
            A tuple of the form (min, max). +ve is CCW
        use_radians : bool
            Are the angles entered in radians or degrees?
        """

        self._swl = swl
        self._dia_pin = dia_pin
        self._out_of_plane_allowance = out_of_plane_allowance
        self._shackle_offset = shackle_offset
        self._out_of_plane_offset = out_of_plane_offset

        if in_plane_angle_limits[0] > in_plane_angle_limits[1]:
            raise ValueError(
                "Minimum allowed in-plane angle is greater than the maximum."
            )

        if not use_radians:
            in_plane_angle_limits = (
                radians(in_plane_angle_limits[0]),
                radians(in_plane_angle_limits[1]),
            )

        self._in_plane_angle_limits = in_plane_angle_limits

        if out_of_plane_angle_limits[0] > in_plane_angle_limits[1]:
            raise ValueError(
                "Minimum allowed out-of-plane angle is greater than the maximum."
            )

        if not use_radians:
            out_of_plane_angle_limits = (
                radians(out_of_plane_angle_limits[0]),
                radians(out_of_plane_angle_limits[1]),
            )

        self._out_of_plane_angle_limits = out_of_plane_angle_limits

    @property
    def swl(self):
        """
        The safe working load of the lug.
        """

        return self._swl

    @property
    def dia_pin(self):
        """
        The diameter of the pin used.
        """

        return self._dia_pin

    @property
    def out_of_plane_allowance(self):
        """
        The out-of-plane allowance of the lug.

        An allowance for any unintended out-of-plane loading.
        Typ. value in AS1418 is 4% of the SWL.
        This is applied in addition to the calculated out-of-plane load.
        """

        return self._out_of_plane_allowance

    @property
    def shackle_offset(self):
        """
        Any offset from the hole centre in the direction towards the load.

        For example, due to the length of the shackle.
        """

        return self._shackle_offset

    @property
    def out_of_plane_offset(self):
        """
        Any horizontal offset out-of-plane.

        For example, if using shackles too wide for the lug.
        """

        return self._out_of_plane_offset

    @property
    def in_plane_angle_limits(self):
        """
        The minimum and maximum angle of the load, in plane.
        """

        return self._in_plane_angle_limits

    @property
    def out_of_plane_angle_limits(self):
        """
        The minimum and maximum angle of the load, out-of-plane.
        """

        return self._out_of_plane_angle_limits

    @property
    def min_in_plane_angle(self):
        """
        The minimum in-plane angle that the lug can be loaded at.
        """

        return self._in_plane_angle_limits[0]

    @property
    def max_in_plane_angle(self):
        """
        The maximum in-plane angle that the lug can be loaded at.
        """

        return self._in_plane_angle_limits[1]

    @property
    def min_out_of_plane_angle(self):
        """
        The minimum out-of-plane angle that the lug can be loaded at.
        """

        return self._out_of_plane_angle_limits[0]

    @property
    def max_out_of_plane_angle(self):
        """
        The maximum out-of-plane angle that the lug can be loaded at.
        """

        return self._out_of_plane_angle_limits[1]

    def generate_single_load(
        self,
        *,
        in_plane_angle,
        out_of_plane_angle,
        match_sign_out_of_plane: bool = True,
    ) -> tuple[float, float]:
        """
        Generate the load components for a single load angle.

        Parameters
        ----------
        in_plane_angle : float
            The angle in plane. +ve is CCW.
        out_of_plane_angle : float
            The angle out-of-plane. +ve is CCW
        match_sign_out_of_plane : bool
            Should the nominal out-of-plane allowance match the
            base out-of-plane load? Typically this should be True
            to generate a conservative load.

        Returns
        -------
        tuple[float, float]
            A tuple with the load and out-of-plane load:
            (in-plane-load, out-of-plane-load).
        """

        if in_plane_angle > self.max_in_plane_angle:
            raise ValueError(
                "in_plane_angle is greater than the maximum allowed: "
                + f"{self.max_in_plane_angle}"
            )
        if in_plane_angle < self.min_in_plane_angle:
            raise ValueError(
                "in_plane_angle is less than the minimum allowed: "
                + "{self.min_in_plane_angle}"
            )
        if out_of_plane_angle > self.max_out_of_plane_angle:
            raise ValueError(
                "out_of_plane_angle is greater than the maximum allowed: "
                + f"{self.max_out_of_plane_angle}"
            )
        if out_of_plane_angle < self.min_out_of_plane_angle:
            raise ValueError(
                "out_of_plane_angle is less than the minimum allowed: "
                + f"{self.min_out_of_plane_angle}"
            )

        sign = (1 if out_of_plane_angle >= 0 else -1) if match_sign_out_of_plane else 1

        return (
            self.swl,
            self.out_of_plane_allowance * sign,
        )

    def in_plane_angles(self, no_increments: int) -> np.ndarray:
        """
        Returns a range of angles between the minimum and maximum in-plane angles.
        Increments are linearly spaced.

        Parameters
        ----------
        no_increments : int
            The no. of increments to return.

        Returns
        -------
        np.ndarray
        """

        if self.min_in_plane_angle == self.max_in_plane_angle:
            return np.asarray([self.min_in_plane_angle])

        return np.linspace(
            self.min_in_plane_angle, self.max_in_plane_angle, no_increments
        )

    def out_of_plane_angles(self, no_increments: int) -> np.ndarray:
        """
        Returns a range of angles between the minimum and maximum out-of-plane angles.
        Increments are linearly spaced.

        Parameters
        ----------
        no_increments : int
            The no. of increments to return.

        Returns
        -------
        np.ndarray
        """

        if self.min_out_of_plane_angle == self.max_out_of_plane_angle:
            return np.asarray([self.min_out_of_plane_angle])

        return np.linspace(
            self.min_out_of_plane_angle, self.max_out_of_plane_angle, no_increments
        )

    def generate_loads(
        self,
        *,
        in_plane_increments: int = 101,
        out_of_plane_increments: int = 21,
        match_sign_out_plane: bool = True,
    ) -> dict[tuple[float, float], tuple[float, float]]:
        """
        Generates loads through a range of angle increments.

        Parameters
        ----------
        in_plane_increments : int
            The number of loads to generate in plane.
        out_of_plane_increments : int
            The number of increments to generate out-of-plane.
        match_sign_out_plane : bool
            Should the nominal out-of-plane allowance match the
            base out-of-plane load? Typically this should be True
            to generate a conservative load.

        Returns
        -------
        dict[tuple[float, float] : tuple[float, float]]
            A dictionary with a Tuple key and tuple value, of the format:
            {(in_plane_angle, out_of_plane_angle): (in-plane-load, out-of-plane-load)}
        """

        in_plane = self.in_plane_angles(no_increments=in_plane_increments)

        out_plane = self.out_of_plane_angles(no_increments=out_of_plane_increments)

        loads = {}

        for i in in_plane:
            for o in out_plane:
                loads[(i, o)] = self.generate_single_load(
                    in_plane_angle=i,
                    out_of_plane_angle=o,
                    match_sign_out_of_plane=match_sign_out_plane,
                )

        return loads

    @property
    def _in_plane_str(self):
        return f"({self.min_in_plane_angle:.3f}, {self.max_in_plane_angle:.3f})"

    @property
    def _out_plane_str(self):
        return f"({self.min_out_of_plane_angle:.3f}, {self.max_out_of_plane_angle:.3f})"

    def __repr__(self):
        return (
            (
                f"{type(self).__name__}("
                + f"{self.swl=:.3f}, "
                + f"{self.out_of_plane_allowance=:.3f}, "
                + f"{self.dia_pin=:.3f}, "
                + f"{self.shackle_offset=:.3f}, "
                + f"{self.out_of_plane_offset=:.3f}, "
                + f"{self._in_plane_str=}, "
                + f"{self._out_plane_str=}), "
            )
            .replace("self._", "")
            .replace("self.", "")
        )


class Lug:
    """
    A class to describe the geometry of a standard lifting lug.
    """

    def __init__(
        self,
        *,
        thickness,
        b_base,
        h_hole,
        radius,
        dia_hole,
        material: SteelGrade,
        e_hole=0.0,
    ):
        """
        Create an object representing a standard lifting lug.

        Parameters
        ----------
        thickness : float
            The thickness of the lug.
        b_base : float
            The length of the base of the lug.
        h_hole : float
            The height from the base to the centre of the hole.
        radius : float
            The radius of the outside of the lug.
        dia_hole : float
            The diameter of the hole.
        material : SteelGrade
            The material to use for the lug.
        e_hole : float
            The eccentricity of the hole from the centre of the base.
        """

        self.thickness = thickness
        self.b_base = b_base
        self.h_hole = h_hole
        self.radius = radius
        self.dia_hole = dia_hole
        self.e_hole = e_hole
        self.material = material

    @property
    def radius_hole(self):
        """
        The radius of the hole.
        """

        return self.dia_hole / 2

    @property
    def x_centre(self):
        """
        The x co-ordinate of the centre of the hole,
        from the centre of the base of the lug.
        """

        return self.e_hole

    @property
    def y_centre(self):
        """
        The y co-ordinate of the centre of the hole,
        from the centre of the base of the lug.
        """
        return self.h_hole

    @property
    def centre(self):
        """
        Return the centre of the circle as a tuple (x, y)
        """

        return self.x_centre, self.y_centre

    @property
    def _l_lhs_ctr(self):
        """
        Distance from the left hand corner to the centre of the lug.
        """

        return ((self.b_base / 2 + self.x_centre) ** 2 + self.y_centre**2) ** 0.5

    @property
    def _theta1_lhs_ctr(self):
        """
        Angle from the LHS base of the lug to the centre of the lug.
        """

        return asin(self.y_centre / self._l_lhs_ctr)

    @property
    def _theta2_lhs_tp1(self):
        """
        The angle between the lines formed by the LHS corner to the tangent point
        and the LHS corner to the centre of the lug.
        """

        return asin(self.radius / self._l_lhs_ctr)

    @property
    def _theta_lhs_tp1(self):
        """
        The angle between tp1 and horizontal, taken from the LHS of the lug
        """

        return self._theta1_lhs_ctr + self._theta2_lhs_tp1

    @property
    def _l_lhs_tp1(self):
        """
        The length of the side of the lug from the LHS corner to tp1.
        """

        return (self._l_lhs_ctr**2 - self.radius**2) ** 0.5

    @property
    def x_tp1(self):
        """
        The x co-ordinate of the LHS tangent point, tp1.
        """

        return self._l_lhs_tp1 * cos(self._theta_lhs_tp1) - self.b_base / 2

    @property
    def y_tp1(self):
        """
        The y co-ordinate of the LHS tangent point, tp1
        """

        return self._l_lhs_tp1 * sin(self._theta_lhs_tp1)

    @property
    def _l_rhs_ctr(self):
        """
        Distance from the RHS corner to the centre of the lug.
        """

        return ((self.b_base / 2 - self.x_centre) ** 2 + self.y_centre**2) ** 0.5

    @property
    def _theta1_rhs_ctr(self):
        """
        Angle from the base of the lug at the RHS to the centre of the lug.
        """
        return asin(self.y_centre / self._l_rhs_ctr)

    @property
    def _theta2_rhs_tp2(self):
        """
        The angle between the lines formed by the RHS corner to the tangent point
        and the LHS corner to the centre of the lug.
        """

        return asin(self.radius / self._l_rhs_ctr)

    @property
    def _theta_rhs_tp2(self):
        """
        The angle between tp2 and horizontal, taken from the RHS of the lug
        """

        return self._theta1_rhs_ctr + self._theta2_rhs_tp2

    @property
    def _l_rhs_tp2(self):
        """
        The length of the side of the lug from the RHS corner to tp2.
        """

        return (self._l_rhs_ctr**2 - self.radius**2) ** 0.5

    @property
    def x_tp2(self):
        """
        The x co-ordinate of the RHS tangent point, tp2.
        """

        return self.b_base / 2 - (self._l_rhs_tp2 * cos(self._theta_rhs_tp2))

    @property
    def y_tp2(self):
        """
        The y co-ordinate of the RHS tangent point, tp2
        """

        return self._l_rhs_tp2 * sin(self._theta_rhs_tp2)

    @property
    def _theta3_tp1(self):
        """
        The horizontal angle from the centre of the circle to the tangent point.
        """

        y_tp1_ctr = self.y_tp1 - self.y_centre
        x_tp1_ctr = self.x_centre - self.x_tp1

        return atan(y_tp1_ctr / x_tp1_ctr)

    @property
    def _theta3_tp2(self):
        """
        The horizontal angle from the centre of the circle to the tangent point.
        """

        y_tp2_ctr = self.y_tp2 - self.y_centre
        x_tp2_ctr = self.x_tp2 - self.x_centre

        return atan(y_tp2_ctr / x_tp2_ctr)

    @property
    def i_xx(self):
        """
        The second moment of inertia about the major axis.
        """

        return self.thickness * self.b_base**3 / 12

    @property
    def i_yy(self):
        """
        The second moment of inertia about the minor axis.
        """

        return self.b_base * self.thickness**3 / 12

    @property
    def base_area(self):
        """
        The area at the base of the lug.
        """

        return self.b_base * self.thickness

    @property
    def face_area(self):
        """
        The face area of the lug.
        """

        return self.lug_polygon().area

    @property
    def volume(self):
        """
        The volume of material in the lug.
        """

        return self.lug_polygon().area * self.thickness

    @property
    def z_major(self):
        """
        Elastic modulus about the major axis.
        """

        return self.thickness * self.b_base**2 / 6

    @property
    def z_minor(self):
        """
        Elastic modulus about the minor axis.
        """

        return self.b_base * self.thickness**2 / 6

    @property
    def s_major(self):
        """
        Plastic modulus about the major axis.
        """

        return self.thickness * self.b_base**2 / 4

    @property
    def s_minor(self):
        """
        Plastic modulus about the minor axis.
        """

        return self.b_base * self.thickness**2 / 4

    @property
    def base_line(self):
        """
        Return a Shapely LineString describing the base of the lug.
        """

        return LineString([(0, 0), (self.b_base, 0)])

    @property
    def total_height(self):
        """
        Return the total height of the lug.
        """

        return self.h_hole + self.radius

    @property
    def f_up(self):
        """
        The ultimate strength of the lug plate.
        """
        return self.material.get_f_u(thickness=self.thickness)

    @property
    def f_yp(self):
        """
        The yield strength of the lug plate.
        """
        return self.material.get_f_y(thickness=self.thickness)

    def resolve_load_about_base(
        self,
        load: float,
        out_of_plane_allowance: float = 0.0,
        angle_in_plane: float = 0.0,
        angle_out_of_plane: float = 0.0,
        hole_offset: float = 0.0,
        out_of_plane_offset: float = 0.0,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Resolves a load into its components.

        Returns a tuple of the form:
            (
                fx: in-plane shear forces.
                fy: vertical forces.
                fz: out-of-plane shear forces.
                mx: bending about the lug's strong axis.
                my: bending about the lug's weak axis.
                mz: torsional loads about the lug.
            )

        Parameters
        ----------
        load : float
            The load to resolve, as a scalar value.
            This should already have any dynamic factors, ULS factors etc. applied.
        out_of_plane_allowance : float
            Any blanket out-of-plane load to apply.
            E.g. AS1418's nominal 4% off-vertical allowance etc.
            This should already have any dynamic factors, ULS factors etc. applied.
            NOTE: The sign of this load is important - this method makes no attempt
            to resolve it into the same direction as the lug load.
        angle_in_plane : float
            The angle of the load in plane, in radians.
            Vertical is 0.0, and CCW is +ve.
        angle_out_of_plane : float
            The angle fo the load out of plane, in radians.
            Vertical is 0.0, and CCW is +ve.
        hole_offset : float
            Any offset in the direction of the load application.
            E.g. if a D-shackle is used that fits tightly to the lug, the load actually
            applies to the top of the D-shackle, not at the hole in the lug.
            This may develop a moment about the weak axis of the lug.
        out_of_plane_offset : float
            Any hole offset in the z direction, such as caused
            by the shackle not being centred on the lug.
        """

        # offsets about the centre of the base.
        ex = self.x_centre - hole_offset * sin(angle_in_plane)
        ey = self.y_centre + hole_offset * cos(angle_in_plane)
        ez = out_of_plane_offset

        # forces about x, y, z
        fx = load * sin(-angle_in_plane) * cos(angle_out_of_plane)
        fy = load * cos(angle_in_plane) * cos(angle_out_of_plane)
        fz = load * -sin(angle_out_of_plane) + out_of_plane_allowance

        # moments about x, y, z
        mx = fx * self.y_centre - fy * self.x_centre
        my = fy * ez - fz * ey
        mz = -fx * ez + fz * ex

        return fx, fy, fz, mx, my, mz

    def _slice(
        self, *, angle: float = 0.0, holes: bool = True, use_radians: bool = True
    ):
        """
        Generate a line describing a slice through the lug.

        Parameters
        ----------
        angle : float
            The angle of the slice, in radians. 0.0rad is vertical.
            +ve angles are CCW
        holes : bool
            Include the hole in the slice?
        use_radians : bool
            use radians or degrees?
        """

        start = (self.x_centre, self.y_centre)
        end = (self.x_centre, self.y_centre + self.total_height)

        line = LineString([start, end])

        line = rotate(
            line,
            angle=angle,
            origin=(self.x_centre, self.y_centre),
            use_radians=use_radians,
        )

        return split(line, self.lug_polygon(holes=holes)).geoms[1]

    def cut(self, *, angle: float = 0.0, holes: bool = True, use_radians: bool = True):
        """
        Return the length of a cut through the lug on a given angle.

        Parameters
        ----------
        angle : float
            The angle of the slice, in radians. 0.0rad is vertical.
            +ve angles are CCW
        holes : bool
            Consider the holes?
            If True, this will be the width of steel between the edge of the hole and
            the edge of the lug. If False, is equivalent to finding the distance from
            the centre of the hole to the edge.
        use_radians : bool
            use radians or degrees?
        """

        return self._slice(angle=angle, holes=holes, use_radians=use_radians).length

    def lug_polygon(self, *, no_points=96, holes: bool = True):
        """
        Return a Shapely polygon describing the lug.

        Parameters
        ----------
        no_points : int
            The no. of points to use for the circular sections of the lug.
        holes : bool
            Include the hole.
        """

        points = [(-self.b_base / 2, 0), (self.b_base / 2, 0)]

        points += build_circle(
            centroid=self.centre,
            radius=self.radius,
            no_points=no_points,
            limit_angles=(self._theta3_tp2, radians(180) - self._theta3_tp1),
        )

        points.append((-self.b_base / 2, 0))

        if not holes:
            return Polygon(shell=points)

        hole_points = build_circle(
            centroid=(self.x_centre, self.y_centre),
            radius=self.dia_hole / 2,
            no_points=no_points,
        )
        hole_points.reverse()

        return Polygon(shell=points, holes=[hole_points])

    def plot_lug(
        self,
        *,
        ax=None,
        face_color=(0.85, 0.85, 0.85, 1.0),
        edge_color=(0, 0, 0),
        add_points=False,
    ):
        """
        Plot the lug. Uses the Shapely plotting library to plot the lug.

        Also returns a tuple of:

        if add_points is False:
            (PathPatch, None)
        else:
            (PathPatch, Line2D)

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axis on which to plot the lug.
        face_color : matplotlib.colors.Color
            A matplotlib color specification to color the face.
        edge_color : matplotlib.colors.Color
            A matplotlib color specification for the edge.
        add_points : bool
            Plot the vertices of the lug?
        """

        if ax is None:
            lug_patch = plot_polygon(
                self.lug_polygon(),
                add_points=add_points,
                facecolor=face_color,
                edgecolor=edge_color,
            )
        else:
            lug_patch = plot_polygon(
                self.lug_polygon(),
                ax=ax,
                add_points=add_points,
                facecolor=face_color,
                edgecolor=edge_color,
            )

        plt.show()

        if add_points is False:
            return lug_patch, None

        return lug_patch

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + f"{self.thickness=:.3f}, "
            + f"{self.b_base=:.3f}, "
            + f"{self.h_hole=:.3f}, "
            + f"{self.radius=:.3f}, "
            + f"{self.dia_hole=:.3f}, "
            + f"{self.e_hole=:.3f})"
        ).replace("self.", "")


def build_circle(
    *,
    centroid: Point | tuple[float, float],
    radius,
    no_points: int = 64,
    limit_angles: tuple[float, float] | None = None,
    use_radians: bool = True,
) -> list[tuple[float, float]]:
    """
    Build a list of points that approximate a circle or circular arc.

    Parameters
    ----------
    centroid : Point | tuple[float, float]
        The centroid of the circle.
    radius : float
        The radius of the circle.
    no_points : int
        The no. of points to include in the definition of the circle.
    limit_angles : tuple[float, float] | None
        Angles to limit the circular arc. Should be of the format
        (min, max). Angles to be taken CCW.
        If None, treated as an implicit (0, 360)
    use_radians : bool
        Use radians for angles?

    Returns
    -------
    list[tuple[float, float]]
        A circle, or part thereof, as a list of lists defining the points:
        [[x1, y1], [x2, y2], ..., [xn, yn]]
    """

    full_circle = radians(360)

    if limit_angles is not None:
        min_angle = limit_angles[0]
        max_angle = limit_angles[1]

        if not use_radians:
            min_angle = radians(min_angle)
            max_angle = radians(max_angle)

    else:
        min_angle = 0
        max_angle = full_circle

    angle_range = np.linspace(start=min_angle, stop=max_angle, num=no_points)
    x_points_orig = np.full(no_points, radius)

    x_points = x_points_orig * np.cos(angle_range)  # - y_points * np.sin(angle_range)
    y_points = x_points_orig * np.sin(angle_range)  # + y_points * np.cos(angle_range)
    # can neglect the 2nd half of the formula because the y-points are just zeroes

    if isinstance(centroid, Point):
        centroid = (centroid.x, centroid.y)

    x_points = x_points + centroid[0]
    y_points = y_points + centroid[1]

    all_points = np.transpose(np.stack((x_points, y_points)))

    return [(p[0], p[1]) for p in all_points.tolist()]
