"""
Calculates basic section properties
"""

import math
from typing import List, Tuple, Union
import abc

from shapely.geometry import Point, Polygon, polygon
import shapely.affinity as aff


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


class GenericSection(Section):
    """
    A generic section that can contain any shape formed from polygons.

    Intended to be used as the base class of any shape formed from a single ring of
    points. Currently not allowed to have holes, holes may be added in the future.
    """

    def __init__(
        self,
        points: Union[List[Point], List[Tuple[float, float]]],
        rotation: float = None,
        rotation_centre: Union[Point, Tuple[float, float]] = None,
        translation: Union[Point, Tuple[float, float]] = None,
    ):
        """
        Initialise a generic section based on an input polygon.

        The polygon can be translated and rotated as part of the initialisation.

        If both translation and rotation are specified, rotation is carried out first,
        followed by translation.

        :param points: The points that make up the polygon. The points do not have to
            be closed. i.e. for points 0, ..., n, if p_0 != p_n the final segment of
            the polygon is assumed to be from point p_n -> p_0.
        :param rotation: An angle value to rotate the polygon about. Angle is in degrees.
        :param rotation_centre: An optional point to complete the rotation about. If
            not given, rotation is about the centroid.
        :param translation: An optional translation to apply to the shape. Note that any
            translation is carried out AFTER rotation.
        """

        poly = polygon.orient(Polygon(points))

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
    def area(self):
        return self.polygon.area

    @property
    def Ixx(self):

        # here we calculate the second moment of inertia via Green's theorem.
        coords = self.polygon.exterior.coords
        I_xx = 0.0

        for i, j in zip(coords[:-1], coords[1:]):

            xi = i[0]
            yi = i[1]
            xj = j[0]
            yj = j[1]

            I_xx += (xj ** 2 + xj * xi + xi ** 2) * (xi * yj - xj * yi)

        return I_xx / 12

    @property
    def Iyy(self):

        coords = self.polygon.exterior.coords
        I_yy = 0.0

        for i, j in zip(coords[:-1], coords[1:]):

            xi = i[0]
            yi = i[1]

            xj = j[0]
            yj = j[1]

            I_yy += (yj ** 2 + yj * yi + yi ** 2) * (xi * yj - xj * yi)

        return I_yy / 12

    @property
    def Ixy(self):

        coords = self.polygon.exterior.coords

        I_xy = 0.0

        for i, j in zip(coords[:-1], coords[1:]):
            xi = i[0]
            yi = i[1]

            xj = j[0]
            yj = j[1]

            I_xy += (2 * xj * yj + xj * yi + xi * yj + 2 * xi * yi) * (
                xi * yj - xj * yi
            )

        return I_xy / 24

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
