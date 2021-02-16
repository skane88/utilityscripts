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
    def x_plus(self) -> float:
        """
        The distance from the centroid of the shape to the most positive extreme x point.
        Note that this should always be a positive quantity - if you need it to be
        negative to correctly determine stresses etc. then account for it appropriately.
        """

        return self.bounding_box[2] - self.x_c

    @property
    def x_minus(self) -> float:
        """
        The distance from the centroid of the shape to the most negative extreme x point.
        Note that this should always be a positive quantity - if you need it to be
        negative to correctly determine stresses etc. then account for it appropriately.
        """

        return self.x_c - self.bounding_box[0]

    @property
    def y_plus(self) -> float:
        """
        The distance from the centroid of the shape to the most positive extreme y point.
        Note that this should always be a positive quantity - if you need it to be
        negative to correctly determine stresses etc. then account for it appropriately.
        """

        return self.bounding_box[3] - self.y_c

    @property
    def y_minus(self) -> float:
        """
        The distance from the centroid of the shape to the most negative extreme y point.
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
    def elastic_modulus_uu_plus(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about an
        axis parallel to the global x-x axis but through the shape's centroid.
        Calculated at the most positive extreme y point.
        """

        return self.Ixx / self.y_plus

    @property
    def elastic_modulus_uu_minus(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about an
        axis parallel to the global x-x axis but through the shape's centroid.
        Calculated at the most negative extreme y point.
        """

        return self.Ixx / self.y_minus

    @property
    def elastic_modulus_vv_plus(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about an
        axis parallel to the global y-y axis but through the shape's centroid.
        Calculated at the most positive extreme x point.
        """

        return self.Iyy / self.x_plus

    @property
    def elastic_modulus_vv_minus(self):
        """
        The elastic section modulus assuming a linear-elastic material behaviour about an
        axis parallel to the global y-y axis but through the shape's centroid.
        Calculated at the most negative extreme x point.
        """

        return self.Iyy / self.x_minus

    @property
    def plastic_modulus_uu(self):
        """
        The plastic section modulus assuming a perfectly plastic material behaviour about
        an axis parallel to the global x-x axis but through the shape's centroid.
        """

        raise NotImplementedError()

    @property
    def plastic_modulus_vv(self):
        """
        The plastic section modulus assuming a perfectly plastic material behaviour about
        an axis parallel to the global y-y axis but through the shape's centroid.
        """

        raise NotImplementedError()

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

    def move(self, x: float, y: float):
        """
        Returns a copy of the object moved by the provided offsets.

        :param x: The x-offset
        :param y: The y offset
        """

        raise NotImplementedError

    def move_to_centre(self):
        """
        Returns a copy of the object moved so that its centroid is at the global origin
        (0,0)
        """

        raise NotImplementedError

    def move_to_point(
        self,
        end_point: Union[Point, Tuple[float, float]],
        origin: Union[str, Point, Tuple[float, float]] = "origin",
    ) -> "Section":
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

        raise NotImplementedError

    def rotate(
        self,
        angle: float,
        origin: Union[str, Point, Tuple[float, float]] = "origin",
        use_radians: bool = True,
    ):
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

        No attempt is made to sort the returned sections by their position relative to the
        split line.

        :param line: The line to split the section on.
        """

        raise NotImplementedError

    def __repr__(self):
        return (
            f"{type(self).__name__}: centroid="
            + f"{self.centroid} "
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
        self, poly: Polygon,
    ):
        """
        Initialise a generic section based on an input polygon.

        :param poly: a shapely polygon object.
        """

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

        return sum([Ixx_from_coords(r) for r in self.coords])

    @property
    def Iyy(self):

        return sum([Iyy_from_coords(r) for r in self.coords])

    @property
    def Ixy(self):

        return sum([Ixy_from_coords(r) for r in self.coords])

    @property
    def centroid(self):

        return self.polygon.centroid

    @property
    def bounding_box(self) -> List[float]:

        return list(self.polygon.bounds)

    def move(self, x: float, y: float):

        return GenericSection(poly=aff.translate(geom=self.polygon, xoff=x, yoff=y))

    def move_to_centre(self):

        c = self.centroid
        return GenericSection(poly=aff.translate(self.polygon, xoff=-c.x, yoff=-c.y))

    def move_to_point(
        self,
        end_point: Union[Point, Tuple[float, float]],
        origin: Union[str, Point, Tuple[float, float]] = "origin",
    ) -> "GenericSection":

        origin = self._make_origin_tuple(origin)

        if isinstance(end_point, Point):
            end_point = (end_point.x, end_point.y)

        xoff = end_point[0] - origin[0]
        yoff = end_point[1] - origin[1]

        return GenericSection(poly=aff.translate(self.polygon, xoff=xoff, yoff=yoff))

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
        A helper function for the split method. This splits the shape up into a number of
        Polygons based on a given split line.

        NOTE: the reason that this one-liner is not included in the actual split method
        is that some other functions require splitting the shape but do not need the
        additional overhead of a Section object.
        (e.g. the plastic section modulus calculators which only need area and centroids)

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

        If no translation or rotation specified, the section centroid is aligned with the
        global origin.

        :param length: The length of the section. By convention aligned with the x-axis
            pre-rotation.
        :param thickness: The thickness of the section. By convention,
            aligned with the y-axis.
        :param rotation_angle: A rotation to apply to the shape.
        :param use_radians: Use radians when rotating or not.
        :param translation: A Tuple containing an (x, y) translation to move the section.
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


class CombinedSection(Section):

    sections: List[Tuple[Section, Point]]

    def __init__(self, sections: List[Tuple[Section, Point]]):
        """
        :param sections: A list of sections & centroids
        """

        all_sections = []
        for s, n in sections:

            if isinstance(s, CombinedSection):

                s: CombinedSection
                for t, o in s.sections:

                    x = n.x + o.x
                    y = n.y + o.y

                    all_sections.append((t, Point(x, y)))

            else:
                all_sections.append((s, n))

        resolved_sections = []

        for s, n in all_sections:
            resolved_sections.append(s.move_to_point(origin="origin", end_point=n))

        for a, b in itertools.combinations(resolved_sections, 2):

            if a.polygon.overlaps(b.polygon):
                # if the intersection is small, we can perhaps ignore it.

                c = a.polygon.intersection(b.polygon)

                if c.area < min(a.area * 0.001, b.area * 0.001):
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

        return sum([s.area for s, n in self.sections])

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

        I_xx = 0

        centroid = self.centroid

        for s, n in self.sections:

            I_xx += s.Iuu + s.area * n.y ** 2

        return I_xx

    @property
    def Iyy(self):

        I_yy = 0

        centroid = self.centroid

        for s, n in self.sections:

            I_yy += s.Iyy + s.area * n.x ** 2

        return I_yy

    @property
    def Ixy(self):

        I_xy = 0

        centroid = self.centroid

        for s, n in self.sections:

            I_xy += s.Ixy + s.area * n.x * n.y

        return I_xy

    @property
    def J_approx(self):

        return sum([s.J_approx for s, n in self.sections])

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

    @property
    def elastic_modulus_uu_plus(self):

        return self.Ixx / self.y_plus

    @property
    def elastic_modulus_uu_minus(self):

        return self.Ixx / self.y_minus

    @property
    def elastic_modulus_vv_plus(self):

        return self.Iyy / self.x_plus

    @property
    def elastic_modulus_vv_minus(self):

        return self.Iyy / self.x_minus

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

    def move_to_centre(self):

        sections = []
        centroid = self.centroid

        for s, n in self.sections:

            offset = Point(n.x - centroid.x, n.y - centroid.y)

            sections.append((s, offset))

        return CombinedSection(sections=sections)

    def move_to_point(
        self,
        origin: Union[str, Point, Tuple[float, float]],
        end_point: Union[Point, Tuple[float, float]],
    ) -> "CombinedSection":

        origin = self._make_origin_tuple(origin)

        if isinstance(end_point, Point):
            end_point = (end_point.x, end_point.y)

        xoff = end_point[0] - origin[0]
        yoff = end_point[1] - origin[1]

        sections = []

        for s, n in self.sections:

            n_point = Point(n.x + xoff, n.y + yoff)

            sections.append((s, n_point))

        return CombinedSection(sections)

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

            # first move the section so that it's centroid accounts for n, so that we can rotate it more easily.
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

    return Rectangle(length=side, height=side)


def make_I(b_f, d, t_f, t_w) -> CombinedSection:
    """
    Make an I section.

    :param b_f: The flange width. May be a single number, or a list of 2x widths
        [top, bottom].
    :param d:
    :param t_f: The flange thickness. May be a single number, or a list of 2x thicknesses,
        [top, bottom].
    :param t_w:
    :return: A CombinedSection representing the I section. Depending on the parameters
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

    n_tf = Point(0, d - t_f_top / 2)
    n_w = Point(0, t_f_bottom + d_w / 2)
    n_bf = Point(0, t_f_bottom / 2)

    return CombinedSection(
        sections=[(top_flange, n_tf), (bottom_flange, n_bf), (web, n_w)]
    ).move_to_centre()


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

    d_w = d - t_f

    flange = Rectangle(length=b_f, thickness=t_f)
    web = Rectangle(length=d_w, thickness=t_w, rotation_angle=90, use_radians=False)

    n_f = Point(0, t_f / 2)
    n_w = Point(0, t_f + d_w / 2)

    T = CombinedSection(sections=[(flange, n_f), (web, n_w)]).move_to_centre()

    if not stem_up:
        T = T.rotate(angle=180, origin="origin", use_radians=False)

    return T


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

    return (I11, I22, alpha)


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

        codes = np.ones(len(ring.coords), dtype=Path.code_type) * Path.LINETO
        codes[0] = Path.MOVETO
        return codes

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
