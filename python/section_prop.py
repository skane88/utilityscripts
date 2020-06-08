"""
Calculates basic section properties
"""

import math
from typing import List, Tuple, Union
import abc

from shapely.geometry import Point, Polygon, polygon
from shapely.coords import CoordinateSequence
import shapely.affinity as aff

import numpy as np
import matplotlib.pyplot as plt


class Section(abc.ABC):
    """
    Parent section class
    """

    @property
    @abc.abstractmethod
    def area(self):
        """
        The cross sectional area of the section.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Ixx(self):
        """
        The second moment of inertia about the GEOMETRIC x-x axis.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Iyy(self):
        """
        The second moment of inertia about the GEOMETRIC y-y axis.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Ixy(self):
        """
        The product of inertia about the GEOMETRIC x, y axes.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Iuu(self):
        """
        The moment of inertia about an axis parallel with the GEOMETRIC x-x axis, but
        through the centroid of the section.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Ivv(self):
        """
        The moment of inertia about an axis parallel with the GEOMETRIC y-y axis, but
        through the centroid of the section.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Iuv(self):
        """
        The product of inertia about the axes parallel with the GEOMETRIC x-x and y-y
        axes, but through the centroid of the section.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def principal_angle(self):
        """
        The principal axis angle.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def J(self):
        """
        The St-Venant's torsional constant of the section.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def centroid(self):
        """
        The location of the centroid of the section.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def depth(self):
        """
        The depth of the section (max(y) - min(y)).
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def width(self):
        """
        The width of the section (max(x) - min(x)).
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def bounding_box(self) -> List[float]:
        """
        The bounding box of the section:

            [min_x, min_y, max_x, max_y]
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def x_and_ys(self) -> List[Tuple[List[float], List[float]]]:
        """
        Return the x and y points that define the section. If the shape or parts of it
        are circular, returns an approximation in straight line segments.

        Given points p_0, ..., p_n defining the section, the method will return:

        [([x_0, ..., x_n, x_0], [y_0, ..., y_n, y_0])]

        The surrounding list is added to allow objects containing multiple sections, or
        sections with holes to return multiple lists of xs and ys.
        """

        raise NotImplementedError

    def plot(self):
        """
        Plot the section using matplotlib.
        """

        for i in self.x_and_ys:
            x, y = i
            plt.plot(x, y)

        plt.show()


class GenericSection(Section):
    """
    A generic section that can contain any shape formed from polygons.

    Intended to be used as the base class of any shape formed from a single ring of
    points. Currently not allowed to have holes, holes may be added in the future.
    """

    def __init__(
        self,
        poly: Polygon,
        rotation: float = None,
        rotation_centre: Union[Point, Tuple[float, float]] = None,
        translation: Union[Point, Tuple[float, float]] = None,
    ):
        """
        Initialise a generic section based on an input polygon.

        The polygon can be translated and rotated as part of the initialisation.

        If both translation and rotation are specified, rotation is carried out first,
        followed by translation.

        :param poly: a shapely polygon object.
        :param rotation: An angle value to rotate the polygon about. Angle is in degrees.
        :param rotation_centre: An optional point to complete the rotation about. If
            not given, rotation is about the centroid.
        :param translation: An optional translation to apply to the shape. Note that any
            translation is carried out AFTER rotation.
        """

        poly = polygon.orient(poly)

        self._input_poly = poly
        self._rotation = rotation
        self._rotation_centre = rotation_centre
        self._translation = translation

        if not rotation is None:

            if rotation_centre is None:
                rotation_centre = "centroid"

            poly = aff.rotate(
                poly, angle=rotation, origin=rotation_centre, use_radians=False
            )

        if not translation is None:

            if isinstance(translation, Point):
                translation = (translation.x, translation.y)

            poly = aff.translate(poly, xoff=translation[0], yoff=translation[1])

        self._polygon = poly

    @property
    def polygon(self) -> Polygon:
        """
        Return the underlying polygon which stores the shape.
        """
        return self._polygon

    @property
    def coords(self):
        """
        Return the coordinates that make up the polygon. Returned as a list of
        coordinate sequences. The first is the exterior polygon, with any additional
        sequences representing interior holes / voids.

        The exterior is ordered CCW, and any holes are ordered CW.
        """

        coords = [self.polygon.exterior.coords]

        for r in self.polygon.interiors:
            coords.append(r.coords)

        return coords

    @property
    def area(self):
        return self.polygon.area

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
    def Iuu(self):

        raise NotImplementedError

    @property
    def Ivv(self):

        raise NotImplementedError

    @property
    def Iuv(self):

        raise NotImplementedError

    @property
    def principal_angle(self):

        raise NotImplementedError

    @property
    def J(self):

        raise NotImplementedError

    @property
    def centroid(self):

        return self.polygon.centroid

    @property
    def depth(self):

        raise NotImplementedError

    @property
    def width(self):

        raise NotImplementedError

    @property
    def bounding_box(self) -> List[float]:

        return self.polygon.bounds

    @property
    def x_and_ys(self):

        retval = []

        rings = [self.polygon.exterior]

        for i in self.polygon.interiors:
            rings.append(i)

        for r in rings:
            x = []
            y = []

            for p in r:
                x.append(p[0])
                y.append(p[1])

            retval.append((x, y))

        return retval

    def move_to_centre(self):

        return


class Rectangle(Section):
    def __init__(self, length, height):

        self.length = length
        self.height = height

    @property
    def area(self):
        return self.length * self.height

    @property
    def Ixx(self):

        return self.length * self.height ** 3 / 12

    @property
    def Iyy(self):

        return self.height * self.length ** 3 / 12

    @property
    def principal_angle(self):

        if self.length > self.height:
            return math.radians(90)

        return 0

    @property
    def J(self):

        t = min(self.length, self.height)
        b = max(self.length, self.height)

        return (b * t ** 3) / 3

    @property
    def centroid(self):

        return Point(self.length / 2, self.height / 2)

    @property
    def depth(self):

        return self.height

    @property
    def width(self):

        return self.length

    @property
    def bounding_box(self) -> List[Point]:

        return [-self.length / 2, -self.height / 2, self.length / 2, self.height / 2]


class CombinedSection(Section):
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

        self.sections = all_sections

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

            I_xx += s.Ixx + s.area * (n.y - centroid.y) ** 2

        return I_xx

    @property
    def Iyy(self):

        I_yy = 0

        centroid = self.centroid

        for s, n in self.sections:

            I_yy += s.Iyy + s.area * (n.x - centroid.x) ** 2

        return I_yy

    @property
    def principal_angle(self):

        raise NotImplementedError

    @property
    def J(self):

        return sum([s.J for s, n in self.sections])

    @property
    def depth(self):

        bbx = self.bounding_box
        return bbx[1].y - bbx[0].y

    @property
    def width(self):

        bbx = self.bounding_box
        return bbx[1].x - bbx[0].x

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

    def add_element(self, section, centroid):

        self.sections.append((section, centroid))

    def move_to_centre(self):

        sections = []
        centroid = self.centroid

        for s, n in self.sections:

            sections.append((s, n - centroid))

        return CombinedSection(sections=sections)


def make_square(side):

    return Rectangle(length=side, height=side)


def make_I(cls, b_f, d, t_f, t_w):

    d_w = d - 2 * t_f

    top_flange = Rectangle(length=b_f, height=t_f)
    bottom_flange = Rectangle(length=b_f, height=t_f)
    web = Rectangle(length=t_w, height=d_w)

    depth = t_f * 2 + d_w

    n_tf = Point(b_f / 2, depth - t_f / 2)
    n_w = Point(b_f / 2, t_f + d_w / 2)
    n_bf = Point(b_f / 2, t_f / 2)

    return cls(
        sections=[(top_flange, n_tf), (bottom_flange, n_bf), (web, n_w)]
    ).move_to_centre()


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


def calculate_principal_moments(Iuu, Ivv, Iuv):
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
