"""
Calculates basic section properties
"""

import math
import itertools
from typing import List, Tuple, Union, TypeVar

from shapely.geometry import Point, Polygon, polygon, LineString
from shapely.coords import CoordinateSequence
import shapely.affinity as aff
import shapely.ops as ops

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from solvers import secant

# an allowance for one section overlapping another in a CombinedSection, as a fraction
# of the smaller section. Set at 0.1% currently.
OVERLAP_TOLERANCE = 0.001

DEFAULT_FACE_COLOR = "#CCCCCC"
DEFAULT_EDGE_COLOR = "#666666"


S = TypeVar("S", bound="Section")


class Section:
    """
    Parent section class
    """

    @property
    def polygon(self) -> Polygon:
        """
        A shapely Polygon that represents the section. For some Sections this may be the
        actual object on which calculations are done (e.g. GenericSection objects) but
        for others it may be merely by a convenient representation for plotting purposes
        etc.
        """

        raise NotImplementedError

    @property
    def area(self):
        """
        The cross sectional area of the section.
        """

        raise NotImplementedError

    @property
    def Ixx(self):
        """
        The second moment of inertia about the GEOMETRIC x-x axis.
        """

        raise NotImplementedError

    @property
    def Iyy(self):
        """
        The second moment of inertia about the GEOMETRIC y-y axis.
        """

        raise NotImplementedError

    @property
    def Izz(self):
        """
        The polar second moment of inertia about the x-x and y-y axes.
        """

        return self.Ixx + self.Iyy

    @property
    def Ixy(self):
        """
        The product of inertia about the GEOMETRIC x, y axes.
        """

        raise NotImplementedError

    @property
    def rxx(self):
        """
        The radius of gyration about the x-x axis.
        """

        return (self.Ixx / self.area) ** 0.5

    @property
    def ryy(self):
        """
        The radius of gyration about the y-y axis.
        """

        return (self.Iyy / self.area) ** 0.5

    @property
    def rzz(self):
        """
        The polar radius of gyration about the x-x & y-y axes.
        """

        return (self.Izz / self.area) ** 0.5

    @property
    def Iuu(self):
        """
        The moment of inertia about an axis parallel with the global x-x axis, but
        through the centroid of the section.
        """

        # note: could be sped up by using the relationship Iuu = Ixx + A*y**2
        # but this loses some accuracy due to floating point operations.
        return self.move_to_centre().Ixx

    @property
    def Ivv(self):
        """
        The moment of inertia about an axis parallel with the global y-y axis, but
        through the centroid of the section.
        """

        # note: could be sped up by using the relationship Iuu = Ixx + A*y**2
        # but this loses some accuracy due to floating point operations.
        return self.move_to_centre().Iyy

    @property
    def Iww(self):
        """
        The polar second moment of inertia about the x-x and y-y axes but through the
        centroid of the section.
        """

        return self.Iuu + self.Ivv

    @property
    def Iuv(self):
        """
        The product of inertia about the axes parallel with the GEOMETRIC x-x and y-y
        axes, but through the centroid of the section.
        """

        # note: could be sped up by using the relationship Iuv = Ixy + A*x*y
        # but this loses some accuracy due to floating point operations.
        return self.move_to_centre().Ixy

    @property
    def ruu(self):
        """
        The radius of gyration about the global x-x axis but through the centroid of the
        section.
        """

        return (self.Iuu / self.area) ** 0.5

    @property
    def rvv(self):
        """
        The radius of gyration about the global y-y axis but through the centroid of the
        section.
        """

        return (self.Ivv / self.area) ** 0.5

    @property
    def rww(self):
        """
        The polar radius of gyration about the global x-x and y-y axes but through the
        centroid of the section.
        """

        return (self.Iww / self.area) ** 0.5

    @property
    def I11(self):
        """
        The major principal moment of inertia.
        """

        return calculate_principal_moments(self.Iuu, self.Ivv, self.Iuv)[0]

    @property
    def I22(self):
        """
        The minor principal moment of inertia.
        """

        return calculate_principal_moments(self.Iuu, self.Ivv, self.Iuv)[1]

    @property
    def I33(self):
        """
        The polar moment of inertia about the principal axes.
        """

        return self.I11 + self.I22

    @property
    def I12(self):
        """
        The product moment of inertia about the principal axes. By definition this is
        always 0.
        """

        return 0.0

    @property
    def r11(self):
        """
        The radius of gyration about the 1-1 principal axis.
        """

        return (self.I11 / self.area) ** 0.5

    @property
    def r22(self):
        """
        The radius of gyration about the 2-2 principal axis.
        """

        return (self.I22 / self.area) ** 0.5

    @property
    def r33(self):
        """
        The polar radius of gyration about the major principal axes.
        """

        return (self.I33 / self.area) ** 0.5

    @property
    def principal_angle(self):
        """
        The principal axis angle in radians.
        """

        return calculate_principal_moments(self.Iuu, self.Ivv, self.Iuv)[2]

    @property
    def principal_angle_degrees(self):
        """
        The principal axis angle in degrees.
        """

        return math.degrees(self.principal_angle)

    @property
    def J(self):
        """
        The St-Venant's torsional constant of the section.
        """

        raise NotImplementedError

    @property
    def Iw(self):
        """
        The warping constant of the section.
        """

        raise NotImplementedError

    @property
    def centroid(self):
        """
        The location of the centroid of the section.
        """

        raise NotImplementedError

    @property
    def x_c(self) -> float:
        """
        The x co-ordinate of the centroid
        """

        return self.centroid.x

    @property
    def y_c(self) -> float:
        """
        The y co-ordinate of the centroid.
        """

        return self.centroid.y

    @property
    def bounding_box(self) -> List[float]:
        """
        The bounding box of the section:

            [min_x, min_y, max_x, max_y]
        """

        raise NotImplementedError

    @property
    def x_min(self) -> float:
        """
        THe minimum x co-ordinate. Equivalent to self.bounding_box[0]
        """

        return self.bounding_box[0]

    @property
    def x_max(self) -> float:
        """
        The maximum x co-ordinate. Equivalent to self.bounding_box[2]
        """

        return self.bounding_box[2]

    @property
    def y_min(self) -> float:
        """
        THe minimum y co-ordinate. Equivalent to self.bounding_box[1]
        """

        return self.bounding_box[1]

    @property
    def y_max(self) -> float:
        """
        The maximum y co-ordinate. Equivalent to self.bounding_box[3]
        """

        return self.bounding_box[3]

    @property
    def extreme_x_plus(self) -> float:
        """
        The distance from the centroid of the shape to the most positive extreme x
        point.
        Note that this should always be a positive quantity - if you need it to be
        negative to correctly determine stresses etc. then account for it appropriately.
        """

        return self.bounding_box[2] - self.x_c

    @property
    def extreme_x_minus(self) -> float:
        """
        The distance from the centroid of the shape to the most negative extreme x
        point.
        Note that this should always be a positive quantity - if you need it to be
        negative to correctly determine stresses etc. then account for it appropriately.
        """

        return self.x_c - self.bounding_box[0]

    @property
    def extreme_y_plus(self) -> float:
        """
        The distance from the centroid of the shape to the most positive extreme y
        point.
        Note that this should always be a positive quantity - if you need it to be
        negative to correctly determine stresses etc. then account for it appropriately.
        """

        return self.bounding_box[3] - self.y_c

    @property
    def extreme_y_minus(self) -> float:
        """
        The distance from the centroid of the shape to the most negative extreme y
        point.
        Note that this should always be a positive quantity - if you need it to be
        negative to correctly determine stresses etc. then account for it appropriately.
        """

        return self.y_c - self.bounding_box[1]

    @property
    def depth(self):
        """
        The overall depth of the section between most extreme y points
        """

        bbx = self.bounding_box
        return bbx[3] - bbx[1]

    @property
    def width(self):
        """
        The overall width of the section between most extreme x points
        :return:
        """

        bbx = self.bounding_box
        return bbx[2] - bbx[0]

    @property
    def extreme_11_plus(self):
        """
        The distance from the centroid of the shape to the most negative extreme 11
        point.
        Note that this should always be a positive quantity - if you need it to be
        negative to correctly determine stresses etc. then account for it appropriately.
        """

        return self.align_to_principal().extreme_x_plus

    @property
    def extreme_11_minus(self):
        """
        The distance from the centroid of the shape to the most negative extreme 11
        point.
        Note that this should always be a positive quantity - if you need it to be
        negative to correctly determine stresses etc. then account for it appropriately.
        """

        return self.align_to_principal().extreme_x_minus

    @property
    def extreme_22_plus(self):
        """
        The distance from the centroid of the shape to the most negative extreme 22
        point.
        Note that this should always be a positive quantity - if you need it to be
        negative to correctly determine stresses etc. then account for it appropriately.
        """

        return self.align_to_principal().extreme_y_plus

    @property
    def extreme_22_minus(self):
        """
        The distance from the centroid of the shape to the most negative extreme 22
        point.
        Note that this should always be a positive quantity - if you need it to be
        negative to correctly determine stresses etc. then account for it appropriately.
        """

        return self.align_to_principal().extreme_y_minus

    @property
    def elastic_modulus_uu_plus(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about
        an axis parallel to the global x-x axis but through the shape's centroid.
        Calculated at the most positive extreme y point.
        """

        return self.Iuu / self.extreme_y_plus

    @property
    def elastic_modulus_uu_minus(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about
        an axis parallel to the global x-x axis but through the shape's centroid.
        Calculated at the most negative extreme y point.
        """

        return self.Iuu / self.extreme_y_minus

    @property
    def elastic_modulus_vv_plus(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about
        an axis parallel to the global y-y axis but through the shape's centroid.
        Calculated at the most positive extreme x point.
        """

        return self.Ivv / self.extreme_x_plus

    @property
    def elastic_modulus_vv_minus(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about
        an axis parallel to the global y-y axis but through the shape's centroid.
        Calculated at the most negative extreme x point.
        """

        return self.Ivv / self.extreme_x_minus

    @property
    def elastic_modulus_uu(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about
        an axis parallel to the global x-x axis but through the shape's centroid.
        """

        return min(self.elastic_modulus_uu_plus, self.elastic_modulus_uu_minus)

    @property
    def elastic_modulus_vv(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about
        an axis parallel to the global y-y axis but through the shape's centroid.
        """

        return min(self.elastic_modulus_vv_plus, self.elastic_modulus_vv_minus)

    @property
    def plastic_modulus_uu(self):
        """
        The plastic section modulus assuming a perfectly plastic material behaviour
        about an axis parallel to the global x-x axis but through the shape's centroid.
        """

        # first calculate the height at which the cut needs to be made

        def helper_func(y):
            """
            Helper function that returns the difference of the area above and below a cut
            line.

            :param y: The height at which to cut the section.
            """

            sections = self.split_horizontal(y_val=y)

            total_area = []

            for s in sections:
                if s.y_c > y:
                    total_area.append(s.area)
                else:
                    total_area.append(-s.area)

            return sum(total_area)

        solution = secant(helper_func, x_low=self.y_min, x_high=self.y_max)
        y = solution[0]

        return self.first_moment_uu(cut_height=y, above=True) + self.first_moment_uu(
            cut_height=y, above=False
        )

    @property
    def plastic_modulus_vv(self):
        """
        The plastic section modulus assuming a perfectly plastic material behaviour
        about an axis parallel to the global y-y axis but through the shape's centroid.
        """

        return self.rotate(angle=90, use_radians=False).plastic_modulus_uu

    @property
    def elastic_modulus_11_plus(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about
        the 11 axis.
        Calculated at the most positive extreme point.
        """

        return self.I11 / self.extreme_22_plus

    @property
    def elastic_modulus_11_minus(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about
        the 11 axis.
        Calculated at the most negative extreme point.
        """

        return self.I11 / self.extreme_22_minus

    @property
    def elastic_modulus_22_plus(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about
        the 22 axis.
        Calculated at the most positive extreme point.
        """

        return self.I22 / self.extreme_11_plus

    @property
    def elastic_modulus_22_minus(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about
        the 22 axis.
        Calculated at the most negative extreme point.
        """

        return self.I22 / self.extreme_11_minus

    @property
    def elastic_modulus_11(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about
        the 11 axis.
        """

        return min(self.elastic_modulus_11_minus, self.elastic_modulus_11_plus)

    @property
    def elastic_modulus_22(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about
        the 22 axis.
        """

        return min(self.elastic_modulus_22_minus, self.elastic_modulus_22_plus)

    @property
    def plastic_modulus_11(self):
        """
        The plastic section modulus assuming a perfectly plastic material behaviour
        about the 11 axis.
        """

        return self.align_to_principal().plastic_modulus_uu

    @property
    def plastic_modulus_22(self):
        """
        The plastic section modulus assuming a perfectly plastic material behaviour
        about the 22 axis.
        """

        return self.align_to_principal().plastic_modulus_vv

    def matplotlib_patches(self, **kwargs):
        """
        Constructs a matplotlib patch of the shape for use in plotting. Relies on the
        section returning a shapely Polygon and the build_patch function.

        :param kwargs: Any valid parameters for the matplotlib.patches.Polygon object.
        """

        return [build_patch(self.polygon, **kwargs)]

    def _find_bounds(self, free_edge: float = 0.25):
        """
        A helper method to find the bounds of the section for plotting purposes.
        Uses the bounding box and expands it to allow for some free space.

        Note that the returned values form a square, not just the original bounding box,
        hence the use of a separate method.

        :param free_edge: how much free space around the edge should be displayed, as a
            fraction of the biggest side of the bounding box.
        """

        bbx = self.bounding_box

        x_range = bbx[2] - bbx[0]
        y_range = bbx[3] - bbx[1]
        max_range = max(x_range, y_range) * (1 + free_edge)

        x_centre = (bbx[2] + bbx[0]) * 0.5
        y_centre = (bbx[3] + bbx[1]) * 0.5

        return (
            (x_centre - max_range / 2),
            (y_centre - max_range / 2),
            (x_centre + max_range / 2),
            (y_centre + max_range / 2),
        )

    def plot(self, **kwargs):
        """
        Plot the section using matplotlib.

        Relies on the section returning a shapely polygon and the build_patch function.

        Method is intended to be over-ridden by Section classes that contain multiple
        polygons etc.

        :param kwargs: Any valid parameters for the matplotlib.patches.Polygon object.
        """

        # set a default format that looks good for cross sections.
        if "fc" not in kwargs and "face_color" not in kwargs:
            kwargs["fc"] = DEFAULT_FACE_COLOR

        if "ec" not in kwargs and "edge_color" not in kwargs:
            kwargs["ec"] = DEFAULT_EDGE_COLOR

        patches = self.matplotlib_patches(**kwargs)

        fig, ax = plt.subplots()

        for p in patches:
            ax.add_patch(p)

        min_x, min_y, max_x, max_y = self._find_bounds()

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect(1.0)

        fig.show()

    def move(self, x: float, y: float) -> S:
        """
        Returns a copy of the object moved by the provided offsets.

        :param x: The x-offset
        :param y: The y offset
        """

        raise NotImplementedError

    def move_to_centre(self) -> S:
        """
        Returns a copy of the object moved so that its centroid is at the global origin
        (0,0)
        """

        return self.move(x=-self.x_c, y=-self.y_c)

    def move_to_point(
        self,
        end_point: Union[Point, Tuple[float, float]],
        origin: Union[str, Point, Tuple[float, float]] = "origin",
    ) -> S:
        """
        Returns a copy of the object translated from the point ``origin`` to the point
        ``end_point``.

        :param end_point: The end point of the move.
        :param origin: The starting point of the movement. Can either be:
            A string: use 'centroid' for the object's geometric centroid,
            'center' for the bounding box center, or 'origin' for the global (0, 0)
            origin.
            A shapely Point object.
            A co-ordinate Tuple (x, y).
        """

        origin = self._make_origin_tuple(origin)

        if isinstance(end_point, Point):
            end_point = (end_point.x, end_point.y)

        x_offset = end_point[0] - origin[0]
        y_offset = end_point[1] - origin[1]

        return self.move(x=x_offset, y=y_offset)

    def rotate(
        self,
        angle: float,
        origin: Union[str, Point, Tuple[float, float]] = "origin",
        use_radians: bool = True,
    ) -> S:
        """
        Returns a copy of the object rotated about a given point.

        :param angle: The angle to rotate. Positive CCW, Negative CW.
        :param origin: The centroid of the rotation. Either provide:
            * A string: use 'centroid' for the object's geometric centroid,
            'center' for the bounding box center, or 'origin' for the global (0, 0)
            origin.
            * A Shapely Point object.
            * A coordinate Tuple (x, y).
        :param use_radians: Is the angle specified in radians or not?
        """

        raise NotImplementedError

    def align_to_principal(self) -> S:
        """
        Return a copy of the shape aligned to the principal axes.
        """

        moved = self.move_to_centre()

        return moved.rotate(angle=-moved.principal_angle)

    def _make_origin_tuple(self, origin):
        """
        Make a Tuple of x, y co-ordinates given an input string or shapely Point

        :param origin: The centroid of the rotation. Either provide:

            A string: use 'centroid' for the object's geometric centroid,
            'center' for the bounding box center, or 'origin' for the global (0, 0)
            origin.
            A Shapely Point object.
        """

        if isinstance(origin, str):
            origin = self._origin_from_string(origin)
        elif isinstance(origin, Point):
            # convert things to tuple for use later
            origin = (origin.x, origin.y)

        return origin

    def _origin_from_string(self, origin):
        """
        Make a Tuple of x, y co-ordinates given an input string or shapely Point

        :param origin: The centroid of the rotation. Either provide:

            A string: use 'centroid' for the object's geometric centroid,
            'center' for the bounding box center, or 'origin' for the global (0, 0)
            origin.
            """

        ctr = self.centroid

        if origin == "center":
            bbx = self.bounding_box
            origin = ((bbx[0] + bbx[2]) / 2, (bbx[3] + bbx[1]) / 2)
        elif origin == "centroid":
            origin = (ctr.x, ctr.y)
        elif origin == "origin":
            origin = (0, 0)

        else:
            raise ValueError(
                f"Expected origin to be either "
                + f"'center', 'centroid' or 'origin', got {origin}"
            )
        return origin

    def split(self, line: LineString) -> List[S]:
        """
        Split the section into two by a line. This method is intended to allow
        the following operations to be implemented:

            * Calculation of first moments of area of a portion of the section.
            * Finding the equal area axis for calculation of plastic section properties.
            * Splitting the section into multiple sections.

        If the line does not cut the section, the original section will be returned.
        If the line cuts the section at least 2x Sections will be returned.
        If the section on one side of the line is non-continuous, it will be returned as
        multiple sections, so there is the potential that there will be more than 2x
        sections returned.

        No attempt is made to sort the returned sections by their position relative to
        the split line.

        :param line: The line to split the section on.
        """

        poly = self.polygon

        if not line.intersects(poly):
            return [self]

        results = ops.split(poly, line)

        return [GenericSection(p) for p in results]

    def split_horizontal(self, y_val) -> List[S]:
        """
        Split the section into two at a given y co-ordinate.
        This method is intended to allow the following operations to be implemented:

            * Calculation of first moments of area of a portion of the section.
            * Finding the equal area axis for calculation of plastic section properties.
            * Splitting the section into multiple sections.

        If the line does not cut the section, the original section will be returned.
        If the line cuts the section at least 2x Sections will be returned.
        If the section on one side of the line is non-continuous, it will be returned as
        multiple sections, so there is the potential that there will be more than 2x
        sections returned.

        No attempt is made to sort the returned sections by their position relative to
        the split line.

        :param y_val: The y-coordinate to split the section at.
        """

        line = LineString([(self.x_min, y_val), (self.x_max, y_val)])

        return self.split(line=line)

    def split_vertical(self, x_val) -> List[S]:
        """
        Split the section into two at a given x co-ordinate.
        This method is intended to allow the following operations to be implemented:

            * Calculation of first moments of area of a portion of the section.
            * Finding the equal area axis for calculation of plastic section properties.
            * Splitting the section into multiple sections.

        If the line does not cut the section, the original section will be returned.
        If the line cuts the section at least 2x Sections will be returned.
        If the section on one side of the line is non-continuous, it will be returned as
        multiple sections, so there is the potential that there will be more than 2x
        sections returned.

        No attempt is made to sort the returned sections by their position relative to
        the split line.

        :param x_val: The x-coordinate to split the section at.
        """

        line = LineString([(x_val, self.y_min), (x_val, self.y_max)])

        return self.split(line=line)

    def _generic_first_moment(self, cut_height, above: bool = True):
        """
        Calculate the generic first moment of a portion of the section above a given cut
        line about the X-X axis

        :param cut_height: The distance above / below the X-X axis to cut the shape.
        :param above: Calculate the first moment of the part of the shape above or below
            the cut?
        """

        line = LineString([(self.x_min, cut_height), (self.x_max, cut_height)])

        broken_section = self.split(line=line)

        first_moment = 0.0

        for s in broken_section:

            if above and s.y_c > cut_height or not above and s.y_c < cut_height:

                first_moment += abs(s.area * s.y_c)

        return first_moment

    def first_moment_uu(self, cut_height, above: bool = True):
        """
        Calculate the generic first moment of a portion of the section above a given cut
        line about the u-u axis

        :param cut_height: The distance above / below the u-u axis to cut the shape.
        :param above: Calculate the first moment of the part of the shape above or below
            the cut?
        """

        # first we move the section so that the centroid lines up with the global origin
        s = self.move_to_centre()

        # now we calculate the first moment
        return s._generic_first_moment(cut_height=cut_height, above=above)

    def first_moment_vv(self, cut_right, right: bool = True):
        """
        Calculate the generic first moment of a portion of the section to the right of a
        given cut line about the
        v-v axis

        :param cut_right: The distance to the right of the axis to cut the shape.
        :param right: Calculate the first moment based on the part to the right or left
            of the cut?
        """

        # first we move the section so that the centroid lines up with the global origin
        s = self.move_to_centre()

        # now we need to rotate it so that "right" is "up"
        s = s.rotate(angle=90, use_radians=False)

        # now return the first moment about u-u.
        return s._generic_first_moment(cut_height=cut_right, above=right)

    def first_moment_11(self, cut_22, above: bool = True):
        """
        Calculate the generic first moment of a portion of the section above a given cut
        line about the 1-1 axis

        :param cut_22: The distance above the 1-1 axis (in the 2-2 direction) to cut the
            shape.
        :param above: Calculate the first moment based on the part above the cut?
        """

        # first we move the section so the centroid lines up with the global origin:
        s = self.align_to_principal()

        # now calculate the first moment
        return s._generic_first_moment(cut_height=cut_22, above=above)

    def first_moment_22(self, cut_11, right: bool = True):
        """
        Calculate the generic first moment of a portion of the section above a given cut
        line about the 2-2 axis

        :param cut_11: The distance to the right of the 2-2 axis (in the 1-1 direction)
            to cut the shape.
        :param right: Calculate the first moment based on the part to the right of the
            cut?
        """

        # first we move the section so the centroid lines up with the global origin:
        s = self.align_to_principal()

        # now rotate by 90 deg so the 2-2 axis is now the 1-1 axis
        s = s.rotate(angle=90, use_radians=False)

        # now calculate the first moment
        return s._generic_first_moment(cut_height=cut_11, above=right)

    def __repr__(self):
        return (
            f"{type(self).__name__}: centroid="
            + f"{self.centroid}"
            + ", bounding box="
            + f"{self.bounding_box}"
            + ", area="
            + f"{self.area}"
        )


class GenericSection(Section):
    """
    A generic section that can contain any shape formed from polygons.

    Intended to be used as the base class of any shape formed from a polygon.
    """

    def __init__(
        self, poly: Union[List[Tuple[float, float]], Polygon],
    ):
        """
        Initialise a generic section based on an input polygon.

        :param poly: a shapely polygon object.
        """

        if isinstance(poly, list):
            poly = Polygon(poly)

        self._polygon = polygon.orient(poly)

    @property
    def polygon(self) -> Polygon:

        return self._polygon

    @property
    def area(self):
        return self.polygon.area

    @property
    def coords(self):
        """
        Return the coordinates that make up the shape.
        """

        return [self.polygon.exterior.coords] + [
            r.coords for r in self.polygon.interiors
        ]

    @property
    def Ixx(self):

        return sum(Ixx_from_coords(r) for r in self.coords)

    @property
    def Iyy(self):

        return sum(Iyy_from_coords(r) for r in self.coords)

    @property
    def Ixy(self):

        return sum(Ixy_from_coords(r) for r in self.coords)

    @property
    def centroid(self):

        return self.polygon.centroid

    @property
    def bounding_box(self) -> List[float]:

        return list(self.polygon.bounds)

    def move(self, x: float, y: float):

        return GenericSection(poly=aff.translate(geom=self.polygon, xoff=x, yoff=y))

    def rotate(
        self,
        angle: float,
        origin: Union[str, Point, Tuple[float, float]] = "origin",
        use_radians: bool = True,
    ):

        origin = self._make_origin_tuple(origin)

        return GenericSection(
            poly=aff.rotate(
                self.polygon, angle=angle, origin=origin, use_radians=use_radians
            )
        )

    def _split_poly(self, line: LineString) -> List[Polygon]:
        """
        A helper function for the split method. This splits the shape up into a number
        of Polygons based on a given split line.

        NOTE: the reason that this one-liner is not included in the actual split method
        is that some other functions require splitting the shape but do not need the
        additional overhead of a Section object.
        (e.g. the plastic section modulus calculators which only need area and
        centroids)

        :param line: The line to split the section on.
        """

        return ops.split(self.polygon, line)

    def split(self, line: LineString) -> List[S]:

        return [GenericSection(p) for p in self._split_poly(line=line)]


class Rectangle(GenericSection):
    def __init__(
        self,
        *,
        length,
        thickness,
        rotation_angle: float = 0,
        use_radians: bool = True,
        translation: Tuple[float, float] = None,
    ):
        """
        A rectangular section. Implemented as a subclass of GenericSection to allow the
        recording of additional properties length & height.

        Unless otherwise specified, any methods inherited from the parent GenericSection
        will NOT preserve the length & height information.

        If no translation or rotation specified, the section centroid is aligned with
        the global origin.

        :param length: The length of the section. By convention aligned with the x-axis
            pre-rotation.
        :param thickness: The thickness of the section. By convention,
            aligned with the y-axis.
        :param rotation_angle: A rotation to apply to the shape.
        :param use_radians: Use radians when rotating or not.
        :param translation: A Tuple containing an (x, y) translation to move the
            section.
            Any translation carried out after rotation.
        """

        self.length = length
        self.thickness = thickness

        x = [-length / 2, length / 2, length / 2, -length / 2]
        y = [-thickness / 2, -thickness / 2, thickness / 2, thickness / 2]

        p = Polygon(zip(x, y))

        if rotation_angle != 0:
            p = aff.rotate(
                geom=p, angle=rotation_angle, origin="centroid", use_radians=use_radians
            )

        if translation is not None:
            p = aff.translate(p, xoff=translation[0], yoff=translation[1])

        super().__init__(p)

    @property
    def J_approx(self):
        """
        St Venant's torsional constant calculated using an approximate method based on
        Roark's stress and Strain, Table 10.7 Section 4.
        """

        thickness = min(self.length, self.thickness)
        length = max(self.length, self.thickness)

        a = length / 2
        b = thickness / 2

        p1 = a * b ** 3
        p2 = 16 / 3
        p3 = 3.36 * (b / a) * (1 - (b ** 4 / (12 * (a ** 4))))

        return p1 * (p2 - p3)

    def move(self, x: float, y: float):

        # to maintain the length & thickness properties through the move some
        # monkey patching needs to go on.

        # TODO Consider if we should delete the Rectangle class altogether and just
        #     replace it with a constructor method, similar to make_I and make_T

        moved = super().move(x=x, y=y)
        rect = Rectangle(length=self.length, thickness=self.thickness)
        rect._polygon = moved.polygon

        return rect

    def rotate(
        self,
        angle: float,
        origin: Union[str, Point, Tuple[float, float]] = "origin",
        use_radians: bool = True,
    ):

        # to maintain the length & thickness properties through the rotation some
        # monkey patching needs to go on.

        rotated = super().rotate(angle=angle, origin=origin, use_radians=use_radians)
        rect = Rectangle(length=self.length, thickness=self.thickness)
        rect._polygon = rotated.polygon

        return rect


class CombinedSection(Section):

    sections: List[Tuple[Section, Point]]

    def __init__(
        self, sections: List[Tuple[Section, Union[Point, Tuple[float, float]]]]
    ):
        """
        :param sections: A list of sections & centroids
        """

        all_sections = []
        for s, n in sections:

            if isinstance(n, Tuple):
                # convert tuples into points
                n = Point(n[0], n[1])

            if isinstance(s, CombinedSection):

                s: CombinedSection
                for t, o in s.sections:

                    x = n.x + o.x
                    y = n.y + o.y

                    all_sections.append((t, Point(x, y)))

            else:
                all_sections.append((s, n))

        resolved_sections = [
            s.move_to_point(origin="origin", end_point=n) for s, n in all_sections
        ]

        for a, b in itertools.combinations(resolved_sections, 2):

            if a.polygon.overlaps(b.polygon):
                # if the intersection is small, we can perhaps ignore it.

                c = a.polygon.intersection(b.polygon)

                if c.area < min(a.area * OVERLAP_TOLERANCE, b.area * OVERLAP_TOLERANCE):
                    continue

                raise ValueError(
                    f"Provided sections overlap each other. Sections are {a} and {b}"
                )

        self.sections = all_sections

    @property
    def no_sections(self):

        return len(self.sections)

    @property
    def polygon(self) -> Polygon:

        all_polys = []

        for s, n in self.sections:

            moved_sect = s.move_to_point(end_point=n)
            all_polys.append(moved_sect.polygon)

        return ops.cascaded_union(all_polys)

    @property
    def area(self):

        return sum(s.area for s, n in self.sections)

    @property
    def centroid(self):

        mx = 0
        my = 0

        for s, n in self.sections:

            mx += s.area * n.x
            my += s.area * n.y

        return Point(mx / self.area, my / self.area)

    @property
    def Ixx(self):

        return sum(s.Iuu + s.area * n.y ** 2 for s, n in self.sections)

    @property
    def Iyy(self):

        return sum(s.Iyy + s.area * n.x ** 2 for s, n in self.sections)

    @property
    def Ixy(self):

        return sum(s.Ixy + s.area * n.x * n.y for s, n in self.sections)

    @property
    def J_approx(self):

        return sum(s.J_approx for s, n in self.sections)

    @property
    def bounding_box(self) -> List[float]:

        test_bounding_box = None

        for s, n in self.sections:

            bbox = s.bounding_box
            bbox[0] += n.x
            bbox[1] += n.y
            bbox[2] += n.x
            bbox[3] += n.y

            if test_bounding_box is None:

                test_bounding_box = bbox

            else:

                test_bounding_box[0] = min(test_bounding_box[0], bbox[0])
                test_bounding_box[1] = min(test_bounding_box[1], bbox[1])
                test_bounding_box[2] = max(test_bounding_box[2], bbox[2])
                test_bounding_box[3] = max(test_bounding_box[3], bbox[3])

        return test_bounding_box

    def generic(self) -> GenericSection:
        """
        Return a generic section of the CombinedSection
        """

        return GenericSection(poly=self.polygon)

    def matplotlib_patches(self, **kwargs):

        """
        Constructs a collection of matplotlib patches of the shape for use in plotting.
        Relies on each underlying section returning a shapely Polygon and the
        build_patch function.

        :param kwargs: Any valid parameters for the matplotlib.patches.Polygon object.
        """

        patches = []

        for s, n in self.sections:

            plot_section = s.move_to_point(origin="origin", end_point=n)

            patches.append(build_patch(plot_section.polygon, **kwargs))

        return patches

    def add_element(self, section, centroid):

        self.sections.append((section, centroid))

    def move(self, x: float, y: float) -> S:
        """
        Returns a copy of the object moved by the provided offsets.

        :param x: The x-offset
        :param y: The y offset
        """

        sections = []

        for s, n in self.sections:

            offset = Point(n.x + x, n.y + y)

            sections.append((s, offset))

        return CombinedSection(sections=sections)

    def rotate(
        self,
        angle: float,
        origin: Union[str, Point, Tuple[float, float]] = "origin",
        use_radians: bool = True,
    ):

        sections = []

        origin = self._make_origin_tuple(origin)

        for s, n in self.sections:

            s: Section
            n: Point

            # first move the section so that it's centroid accounts for n, so that we
            # can rotate it more easily.
            s = s.move_to_point(origin="origin", end_point=n)

            # now rotate it
            s = s.rotate(angle=angle, origin=origin, use_radians=use_radians)

            # now move it back to it's original co-ordinates
            new_centroid = s.centroid
            s = s.move_to_centre()

            sections.append((s, new_centroid))

        return CombinedSection(sections=sections)

    def __repr__(self):

        super_repr = super().__repr__()

        return super_repr + f", no. sections={self.no_sections}"


def make_square(side):

    return Rectangle(length=side, thickness=side)


def make_I(
    b_f,
    d,
    t_f,
    t_w,
    radius_or_weld: str = None,
    radius_size=None,
    weld_size=None,
    box_in: bool = False,
    t_box=None,
) -> CombinedSection:
    """
    Make an I section. if a radius

    :param b_f: The flange width. May be a single number, or a list of 2x widths
        [top, bottom].
    :param d: The total depth of the section.
    :param t_f: The flange thickness. May be a single number, or a list of 2x
        thicknesses, [top, bottom].
    :param t_w: The web thickness.
    :param radius_or_weld: Use 'r' or 'w' as appropriate, or None if ignoring.
    :param radius_size: The size of any radius. Only used if radius_or_weld is 'r'
    :param weld_size: The size of any weld. Only used if radius_or_weld is 'w'
    :param box_in: Box the section in?
    :param t_box: The thickness of any boxing.
    :return: A CombinedSection representing the I section. Depending on the parameters
        entered this may be a collection of rectangular plates or perhaps only a single
        GenericSection
    """

    if isinstance(b_f, (list, tuple)):
        b_f_top, b_f_bottom = b_f
    else:
        b_f_top = b_f
        b_f_bottom = b_f

    if isinstance(t_f, (list, tuple)):
        t_f_top, t_f_bottom = t_f
    else:
        t_f_top = t_f
        t_f_bottom = t_f

    d_w = d - (t_f_top + t_f_bottom)

    if radius_or_weld is None:
        i_sections = _make_I_simple(b_f_bottom, b_f_top, d, t_f_bottom, t_f_top, t_w)

    else:

        if radius_or_weld not in ["r", "w"]:
            raise ValueError("Expected either 'r' or 'w' to define radii or welds.")

        # prepare radii or welds + web
        if radius_or_weld == "r":

            i_sections = _make_I_radius(
                b_f_bottom, b_f_top, d, radius_size, t_f_bottom, t_f_top, t_w
            )

        else:

            i_sections = _make_I_weld(
                b_f_bottom, b_f_top, d, t_f_bottom, t_f_top, t_w, weld_size
            )

    if box_in:

        box_plate = Rectangle(
            length=d_w, thickness=t_box, rotation_angle=90, use_radians=False
        )

        b_f_min = min(b_f_top, b_f_bottom)

        n_box_1 = Point(-b_f_min / 2 + t_box / 2, t_f_bottom + d_w / 2)
        n_box_2 = Point(b_f_min / 2 - t_box / 2, t_f_bottom + d_w / 2)

        i_sections.append((box_plate, n_box_1))
        i_sections.append((box_plate, n_box_2))

    return CombinedSection(sections=i_sections).move_to_centre()


def _make_I_weld(b_f_bottom, b_f_top, d, t_f_bottom, t_f_top, t_w, weld_size):
    """
    Make an I section with welds between web and flange.

    :param b_f_bottom: Bottom flange width.
    :param b_f_top: Top flange width.
    :param d: Depth of the section.
    :param t_f_bottom: Thickness of bottom flange.
    :param t_f_top: Thickness of top flange.
    :param t_w: Thickness of web.
    :param weld_size: Weld size.
    """

    if weld_size is None:
        raise ValueError(f"Expected a weld size, got {weld_size}")

    if isinstance(weld_size, list):
        w_top = weld_size[0]
        w_bottom = weld_size[1]
    else:
        w_top = weld_size
        w_bottom = weld_size

    # make the flanges
    bottom_flange = [
        [-b_f_bottom / 2, t_f_bottom],
        [-b_f_bottom / 2, 0],
        [b_f_bottom / 2, 0],
        [b_f_bottom / 2, t_f_bottom],
    ]
    top_flange = [
        [b_f_top / 2, d - t_f_top],
        [b_f_top / 2, d],
        [-b_f_top / 2, d],
        [-b_f_top / 2, d - t_f_top],
    ]
    close_flange = [[-b_f_bottom / 2, t_f_bottom]]

    # now make the welds
    bottom_right = [
        [t_w / 2 + w_bottom, t_f_bottom],
        [t_w / 2, t_f_bottom + w_bottom],
    ]
    top_right = [
        [t_w / 2, d - t_f_top - w_top],
        [t_w / 2 + w_top, d - t_f_top],
    ]
    top_left = [
        [-t_w / 2 - w_top, d - t_f_top],
        [-t_w / 2, d - t_f_top - w_top],
    ]
    bottom_left = [
        [-t_w / 2, t_f_bottom + w_bottom],
        [-t_w / 2 - w_bottom, t_f_bottom],
    ]

    # now assemble the polygon that will be used to make the section.
    points = (
        bottom_flange
        + bottom_right
        + top_right
        + top_flange
        + top_left
        + bottom_left
        + close_flange
    )

    poly = Polygon(points)

    # now make the section.
    i_section = GenericSection(poly).move_to_centre()
    return [(i_section, Point(0, abs(i_section.y_min)))]


def _make_I_radius(b_f_bottom, b_f_top, d, radius_size, t_f_bottom, t_f_top, t_w):
    """
    Make an I section with a radius between web and flange.

    :param b_f_bottom: Bottom flange thickness.
    :param b_f_top: Top flange thickness.
    :param d: Web depth.
    :param radius_size: Radius between web and flanges.
    :param t_f_bottom: Thickness of bottom flange.
    :param t_f_top: Thickness of top flange.
    :param t_w: Thickness of the web.
    """

    if radius_size is None:
        raise ValueError(f"Expected a radius, got {radius_size}.")
    if isinstance(radius_size, list):
        r_top = radius_size[0]
        r_bottom = radius_size[1]
    else:
        r_top = radius_size
        r_bottom = radius_size

    # make the flanges
    bottom_flange = [
        [-b_f_bottom / 2, t_f_bottom],
        [-b_f_bottom / 2, 0],
        [b_f_bottom / 2, 0],
        [b_f_bottom / 2, t_f_bottom],
    ]
    top_flange = [
        [b_f_top / 2, d - t_f_top],
        [b_f_top / 2, d],
        [-b_f_top / 2, d],
        [-b_f_top / 2, d - t_f_top],
    ]
    close_flange = [[-b_f_bottom / 2, t_f_bottom]]

    # now generate the radii
    bottom_right = list(
        reversed(
            build_circle(
                centroid=(t_w / 2 + r_bottom, t_f_bottom + r_bottom,),
                radius=r_bottom,
                angles=(180, 270),
                use_radians=False,
                no_points=16,
            )
        )
    )
    top_right = list(
        reversed(
            build_circle(
                centroid=(t_w / 2 + r_top, d - t_f_top - r_top),
                radius=r_top,
                angles=(90, 180),
                use_radians=False,
                no_points=16,
            )
        )
    )
    top_left = list(
        reversed(
            build_circle(
                centroid=(-t_w / 2 - r_top, d - t_f_top - r_top),
                radius=r_top,
                angles=(0, 90),
                use_radians=False,
                no_points=16,
            )
        )
    )
    bottom_left = list(
        reversed(
            build_circle(
                centroid=(-t_w / 2 - r_bottom, t_f_bottom + r_bottom,),
                radius=r_bottom,
                angles=(270, 360),
                use_radians=False,
                no_points=16,
            )
        )
    )

    # now assemble the polygon that will be used to make the section.
    points = (
        bottom_flange
        + bottom_right
        + top_right
        + top_flange
        + top_left
        + bottom_left
        + close_flange
    )

    poly = Polygon(points)

    # now make the section.
    i = GenericSection(poly).move_to_centre()
    return [(i, Point(0, abs(i.y_min)))]


def _make_I_simple(b_f_bottom, b_f_top, d, t_f_bottom, t_f_top, t_w):
    """
    Make a simple I section out of 3x rectangular sections.

    :param b_f_bottom: The bottom flange width.
    :param b_f_top: The top flange width.
    :param d: The depth of the section.
    :param t_f_bottom: The thickness of the bottom flange.
    :param t_f_top: The thickness of the top flange.
    :param t_w: The web thickness.
    """

    # calculate the web depth
    d_w = d - (t_f_top + t_f_bottom)

    # 3x rectangular sections
    top_flange = Rectangle(length=b_f_top, thickness=t_f_top)
    bottom_flange = Rectangle(length=b_f_bottom, thickness=t_f_bottom)
    web = Rectangle(length=d_w, thickness=t_w, rotation_angle=90, use_radians=False)

    # 3x centroids
    n_tf = Point(0, d - t_f_top / 2)
    n_w = Point(0, t_f_bottom + d_w / 2)
    n_bf = Point(0, t_f_bottom / 2)

    return [(top_flange, n_tf), (bottom_flange, n_bf), (web, n_w)]


def make_T(b_f, d, t_f, t_w, stem_up: bool = True) -> CombinedSection:
    """
    A helper method to make a T section.

    :param b_f: The flange width.
    :param d: The total depth of the section.
    :param t_f: The flange thickness.
    :param t_w: The web thickness.
    :param stem_up: Is the T-stem up or down?
    :return: A CombinedSection representing the T section. Depending on the parameters
        entered this may be a collection of rectangular plates or perhaps only a single
        GenericSection
    """

    t = _make_T_simple(b_f, d, t_f, t_w)

    if not stem_up:
        t = t.rotate(angle=180, origin="origin", use_radians=False)

    return t


def _make_T_simple(b_f, d, t_f, t_w):
    """
    Make a simple T-section from Rectangle objects. T stem faces up.

    :param b_f: Flange width.
    :param d: Depth of section.
    :param t_f: Thickness of flange.
    :param t_w: Thickness of web.
    """

    # calculate web depth
    d_w = d - t_f

    # create flange and web.
    flange = Rectangle(length=b_f, thickness=t_f)
    web = Rectangle(length=d_w, thickness=t_w, rotation_angle=90, use_radians=False)

    # centroids of sections.
    n_f = Point(0, t_f / 2)
    n_w = Point(0, t_f + d_w / 2)

    return CombinedSection(sections=[(flange, n_f), (web, n_w)]).move_to_centre()


def make_C(b_f, d, t_f, t_w, box_in: bool = False, t_box=None) -> CombinedSection:
    """
    Make a C section. By default it is orientated with its toes to the right.

    :param b_f: The flange width. May be a single number, or a list of 2x widths
        [top, bottom].
    :param d: The total depth of the section.
    :param t_f: The flange thickness. May be a single number, or a list of 2x
        thicknesses, [top, bottom].
    :param t_w: The web thickness.
    :param box_in: Box the section in?
    :param t_box: The thickness of any boxing.
    :return: A CombinedSection representing the C section. Depending on the parameters
        entered this may be a collection of rectangular plates or perhaps only a single
        GenericSection
    """

    if isinstance(b_f, list):
        b_f_top = b_f[0]
        b_f_bottom = b_f[1]
    else:
        b_f_top = b_f
        b_f_bottom = b_f

    if isinstance(t_f, list):
        t_f_top = t_f[0]
        t_f_bottom = t_f[1]
    else:
        t_f_top = t_f
        t_f_bottom = t_f

    d_w = d - (t_f_top + t_f_bottom)

    top_flange = Rectangle(length=b_f_top, thickness=t_f_top)
    bottom_flange = Rectangle(length=b_f_bottom, thickness=t_f_bottom)
    web = Rectangle(length=d_w, thickness=t_w, rotation_angle=90, use_radians=False)

    n_tf = Point(b_f / 2, d - t_f_top / 2)
    n_w = Point(t_w / 2, t_f_bottom + d_w / 2)
    n_bf = Point(b_f / 2, t_f_bottom / 2)

    c_sections = [(top_flange, n_tf), (bottom_flange, n_bf), (web, n_w)]

    if box_in:

        box_plate = Rectangle(
            length=d_w, thickness=t_box, rotation_angle=90, use_radians=False
        )

        n_box = Point(b_f - t_box / 2, t_f_bottom + d_w / 2)

        c_sections.append((box_plate, n_box))

    return CombinedSection(sections=c_sections).move_to_centre()


def _prepare_coords_for_green(
    coords: Union[CoordinateSequence, List[Tuple[float, float]], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares a set of co-ordinates for use in Green's integration algorithms.
    :param coords: The coordinates of the object as a Shapely CoordinateSequence or
        an equivalent 2D array of coordinates (x, y vertically orientated).

        The co-ordinates should be closed - that is, given a sequence of points
        p_0, ..., p_n making up a shape, the co-ordinates for the object provided to
        this method should end with p_0: p_0, ..., p_n, p_0.

        Points should be ordered counterclockwise for positive quantities. Holes should
        be ordered clockwise.
    """

    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)

    # in case we get a numpy array that is orientated with x & y as rows, not columns,
    # transpose it to match the expected output from a Coordinate sequence or a list of
    # point tuples
    if coords.shape[0] < coords.shape[1]:
        coords = coords.transpose()

    # get 1D arrays of the x and y coordinates
    xi = coords[:-1, :1]
    yi = coords[:-1, 1:]
    xj = coords[1:, :1]
    yj = coords[1:, 1:]

    return xi, xj, yi, yj


def Ixx_from_coords(
    coords: Union[CoordinateSequence, List[Tuple[float, float]], np.ndarray]
) -> float:
    """
    Calculate the moment of inertia about the global x axis by Green's theorem.

    :param coords: The coordinates of the object as a Shapely CoordinateSequence or
        an equivalent 2D array of coordinates (x, y vertically orientated).

        The co-ordinates should be closed - that is, given a sequence of points
        p_0, ..., p_n making up a shape, the co-ordinates for the object provided to
        this method should end with p_0: p_0, ..., p_n, p_0.

        Points should be ordered counterclockwise for positive quantities. Holes should
        be ordered clockwise.
    :return: The moment of inertia
    """

    xi, xj, yi, yj = _prepare_coords_for_green(coords)

    # carry out Green's integration and return
    return np.sum((yj ** 2 + yj * yi + yi ** 2) * (xi * yj - xj * yi)) / 12


def Iyy_from_coords(
    coords: Union[CoordinateSequence, List[Tuple[float, float]], np.ndarray]
) -> float:
    """
    Calculate the moment of inertia about the global y axis by Green's theorem

    :param coords: The coordinates of the object as a Shapely CoordinateSequence or
        an equivalent 2D array of coordinates (x, y vertically orientated).

        The co-ordinates should be closed - that is, given a sequence of points
        p_0, ..., p_n making up a shape, the co-ordinates for the object provided to
        this method should end with p_0: p_0, ..., p_n, p_0.

        Points should be ordered counterclockwise for positive quantities. Holes should
        be ordered clockwise.
    """

    xi, xj, yi, yj = _prepare_coords_for_green(coords)

    # carry out Green's integration and return
    return np.sum((xj ** 2 + xj * xi + xi ** 2) * (xi * yj - xj * yi)) / 12


def Ixy_from_coords(coords):
    """
    Calculate the product of inertia about the global x-x and y-y axes by
    Green's theorem

    :param coords: The coordinates of the object as a Shapely CoordinateSequence or
        an equivalent 2D array of coordinates (x, y vertically orientated).

        The co-ordinates should be closed - that is, given a sequence of points
        p_0, ..., p_n making up a shape, the co-ordinates for the object provided to
        this method should end with p_0: p_0, ..., p_n, p_0.

        Points should be ordered counterclockwise for positive quantities. Holes should
        be ordered clockwise.
    """

    xi, xj, yi, yj = _prepare_coords_for_green(coords)

    return (
        np.sum((2 * xj * yj + xj * yi + xi * yj + 2 * xi * yi) * (xi * yj - xj * yi))
        / 24
    )


def calculate_principal_moments(
    Iuu: float, Ivv: float, Iuv: float
) -> Tuple[float, float, float]:
    """
    Calculates the principal moments of inertia and their axis given the moments of
    inertia about 2x other axes and the product of inertia.

    Note that Iuu, Ivv and Iuv must be orthogonal, and through the centroid of the
    section.

    u-u and v-v are the orthogonal axes.

    Based on the equation at the following location, also see Roark's Stress & Strain

    https://leancrew.com/all-this/2018/01/transforming-section-properties-and-principal-directions/

    :param Iuu: The moment of inertia about axis uu.
    :param Ivv: The moment of inertia about axis vv.
    :param Iuv: The product of inertia.
    :return: A tuple containing the principal axes and the angle between uu and I11:

        (I11, I22, alpha)
    """

    avg = (Iuu + Ivv) / 2
    diff = (Iuu - Ivv) / 2  # Note that this is signed
    I11 = avg + math.sqrt(diff ** 2 + Iuv ** 2)
    I22 = avg - math.sqrt(diff ** 2 + Iuv ** 2)
    alpha = math.atan2(-Iuv, diff) / 2

    return I11, I22, alpha


def build_path(poly: Polygon) -> Path:
    """
    Builds a matplotlib Path that describes a shapely Polygon.

    Method inspired by the descartes library, but implemented independently here to
    minimise dependencies. Also see:

    https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
    https://bitbucket.org/sgillies/descartes/src/default/descartes/patch.py

    :param poly: A shapely Polygon object, with or without holes. Note that the object
        must be orientated with exterior rings CCW and interior rings CW.
    :return: A matplotlib Path object.
    """

    def get_codes(ring) -> np.ndarray:
        """
        Get the path codes for a coordinate ring.

        These codes will all be "LINETO" except for a "MOVETO" at the start.

        :param ring: A coordinate ring.
        :return: A numpy array containing the path codes.
        """

        path_codes = np.ones(len(ring.coords), dtype=Path.code_type) * Path.LINETO
        path_codes[0] = Path.MOVETO
        return path_codes

    # build a numpy array of the vertices
    vertices = np.concatenate(
        [np.asarray(poly.exterior)] + [np.asarray(c) for c in poly.interiors]
    )
    codes = np.concatenate(
        [get_codes(poly.exterior)] + [get_codes(c) for c in poly.interiors]
    )

    return Path(vertices, codes)


def build_patch(poly: Polygon, **kwargs):
    """
    Constructs a matplotlib patch from a shapely Polygon.


    Method inspired by the descartes library, but implemented independently here to
    minimise dependencies. Also see:

    https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
    https://bitbucket.org/sgillies/descartes/src/default/descartes/patch.py

    :param poly: A shapely Polygon object, with or without holes. Note that the object
        must be orientated with exterior rings CCW and interior rings CW.
    :param kwargs: Any acceptable kwargs for the matplotlib.patches.Polygon class.
    :return: A matplotlib PathPatch describing the polygon
    """

    return PathPatch(build_path(poly), **kwargs)


def build_circle(
    *,
    centroid=Union[Point, Tuple[float, float]],
    radius,
    no_points: int = 64,
    angles: Tuple[float, float] = None,
    use_radians: bool = True,
) -> List[Tuple[float, float]]:
    """
    Build a list of points that approximate a circle or circular arc.

    :param centroid: The centroid of the circle.
    :param radius: The radius of the circle.
    :param no_points: The no. of points to include in the definition of the circle.
    :param angles: Angles to limit the circular arc. Should be of the format (min, max).
        Angles to be taken CCW.
    :param use_radians: Use radians for angles?
    :return: A circle, or part thereof, as a list of lists defining the points:
        [[x1, y1], [x2, y2], ..., [xn, yn]]
    """

    full_circle = math.radians(360)

    if angles is not None:

        min_angle = angles[0]
        max_angle = angles[1]

        if not use_radians:
            min_angle = math.radians(min_angle)
            max_angle = math.radians(max_angle)

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
