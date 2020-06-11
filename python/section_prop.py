"""
Calculates basic section properties
"""

import math
from typing import List, Tuple, Union
import abc

from shapely.geometry import Point, Polygon, polygon, MultiPoint
from shapely.coords import CoordinateSequence
import shapely.affinity as aff

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path


class Section(abc.ABC):
    """
    Parent section class
    """

    @property
    @abc.abstractmethod
    def polygon(self) -> Polygon:
        """
        A shapely Polygon that represents the section.
        """

        raise NotImplementedError

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
    def Izz(self):
        """
        The polar second moment of inertia about the x-x and y-y axes.
        """

        return self.Ixx + self.Iyy

    @property
    @abc.abstractmethod
    def Ixy(self):
        """
        The product of inertia about the GEOMETRIC x, y axes.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def rxx(self):
        """
        The radius of gyration about the x-x axis.
        """

        return (self.Ixx / self.area) ** 0.5

    @property
    @abc.abstractmethod
    def ryy(self):
        """
        The radius of gyration about the y-y axis.
        """

        return (self.Iyy / self.area) ** 0.5

    @property
    @abc.abstractmethod
    def rzz(self):
        """
        The polar radius of gyration about the x-x & y-y axes.
        """

        return (self.Izz / self.area) ** 0.5

    @property
    @abc.abstractmethod
    def Iuu(self):
        """
        The moment of inertia about an axis parallel with the global x-x axis, but
        through the centroid of the section.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Ivv(self):
        """
        The moment of inertia about an axis parallel with the global y-y axis, but
        through the centroid of the section.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Iww(self):
        """
        The polar second moment of inertia about the x-x and y-y axes but through the
        centroid of the section.
        """

        return self.Iuu + self.Ivv

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
    def ruu(self):
        """
        The radius of gyration about the global x-x axis but through the centroid of the
        section.
        """

        return (self.Iuu / self.area) ** 0.5

    @property
    @abc.abstractmethod
    def rvv(self):
        """
        The radius of gyration about the global y-y axis but through the centroid of the
        section.
        """

        return (self.Ivv / self.area) ** 0.5

    @property
    @abc.abstractmethod
    def rww(self):
        """
        The polar radius of gyration about the global x-x and y-y axes but through the
        centroid of the section.
        """

        return (self.Iww / self.area) ** 0.5

    @property
    @abc.abstractmethod
    def I11(self):
        """
        The major principal moment of inertia.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def I22(self):
        """
        The minor principal moment of inertia.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def I33(self):
        """
        The polar moment of inertia about the principal axes.
        """

        return self.I11 + self.I22

    @property
    @abc.abstractmethod
    def I12(self):
        """
        The product moment of inertia about the principal axes. By definition this is
        always 0.
        """

        return 0.0

    @property
    @abc.abstractmethod
    def r11(self):
        """
        The radius of gyration about the 1-1 principal axis.
        """

        return (self.I11 / self.area) ** 0.5

    @property
    @abc.abstractmethod
    def r22(self):
        """
        The radius of gyration about the 2-2 principal axis.
        """

        return (self.I22 / self.area) ** 0.5

    @property
    @abc.abstractmethod
    def r33(self):
        """
        The polar radius of gyration about the major principal axes.
        """

        return (self.I33 / self.area) ** 0.5

    @property
    @abc.abstractmethod
    def principal_angle(self):
        """
        The principal axis angle in radians.
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
    def Iw(self):
        """
        The warping constant of the section.
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
    def bounding_box(self) -> List[float]:
        """
        The bounding box of the section:

            [min_x, min_y, max_x, max_y]
        """

        raise NotImplementedError

    @abc.abstractmethod
    def plot(self, face_color: str = "#cccccc", edge_color: str = "black", **kwargs):
        """
        Plot the section using matplotlib.

        Relies on the section returning a shapely polygon and the build_patch method.

        Method is intended to be over-ridden by Section classes that contain multiple
        polygons etc.

        :param face_color: The face color of the section.
        :param edge_color: The edge color of the section.
        :param kwargs: Any valid parameters for the matplotlib.patches.Polygon object.
        """

        patch = build_patch(self.polygon, fc=face_color, ec=edge_color, **kwargs)

        fig, ax = plt.subplots()
        ax.add_patch(patch)

        bbx = self.bounding_box

        x = bbx[2] - bbx[0]
        y = bbx[3] - bbx[1]

        min_x = bbx[0] - x / 4
        max_x = bbx[2] + x / 4
        min_y = bbx[1] - y / 4
        max_y = bbx[3] + y / 4

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect(1.0)

        fig.show()

    @abc.abstractmethod
    def move_to_centre(self):
        """
        Returns a copy of the object moved so that it's centroid is at the global origin
        (0,0)
        """

        raise NotImplementedError


class GenericSection(Section):
    """
    A generic section that can contain any shape formed from polygons.

    Intended to be used as the base class of any shape formed from a polygon.
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
        :param rotation: An angle value to rotate the polygon about.
            Angle is in radians.
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
                poly, angle=rotation, origin=rotation_centre, use_radians=True
            )

        if not translation is None:

            if isinstance(translation, Point):
                translation = (translation.x, translation.y)

            poly = aff.translate(poly, xoff=translation[0], yoff=translation[1])

        self._polygon = poly

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
    def Izz(self):

        return super().Izz

    @property
    def Ixy(self):

        return sum([Ixy_from_coords(r) for r in self.coords])

    @property
    def rxx(self):

        return super().rxx

    @property
    def ryy(self):

        return super().ryy

    @property
    def rzz(self):

        return super().rzz

    @property
    def Iuu(self):

        # note: could be sped up by using the relationship Iuu = Ixx + A*y**2
        # but this loses some accuracy due to floating point operations.
        return self.move_to_centre().Ixx

    @property
    def Ivv(self):

        # note: could be sped up by using the relationship Ivv = Iyy + A*x**2
        # but this loses some accuracy due to floating point operations.
        return self.move_to_centre().Iyy

    @property
    def Iww(self):

        return super().Iww

    @property
    def Iuv(self):

        # note: could be sped up by using the relationship Iuv = Ixy + A*x*y
        # but this loses some accuracy due to floating point operations.
        return self.move_to_centre().Ixy

    @property
    def ruu(self):

        return super().ruu

    @property
    def rvv(self):

        return super().rvv

    @property
    def rww(self):

        return super().rww

    @property
    def I11(self):

        return calculate_principal_moments(self.Ixx, self.Iyy, self.Ixy)[0]

    @property
    def I22(self):

        return calculate_principal_moments(self.Ixx, self.Iyy, self.Ixy)[1]

    @property
    def I33(self):

        return super().I33

    @property
    def I12(self):

        return super().I12

    @property
    def r11(self):

        return super().r11

    @property
    def r22(self):

        return super().r22

    @property
    def r33(self):

        return super().r33

    @property
    def principal_angle(self):

        return calculate_principal_moments(self.Ixx, self.Iyy, self.Ixy)[2]

    @property
    def J(self):

        raise NotImplementedError

    @property
    def Iw(self):

        raise NotImplementedError

    @property
    def centroid(self):

        return self.polygon.centroid

    @property
    def bounding_box(self) -> List[float]:

        return self.polygon.bounds

    def plot(self, face_color: str = "#cccccc", edge_color: str = "black", **kwargs):

        super().plot(face_color=face_color, edge_color=edge_color, **kwargs)

    def move_to_centre(self):

        c = self.centroid
        return GenericSection(poly=aff.translate(self.polygon, xoff=-c.x, yoff=-c.y))


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
