"""
Calculates basic section properties
"""

import math
from typing import List, Tuple
import abc

from shapely.geometry import Point, Polygon


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
    def I_xx(self):
        """
        The second moment of inertia about the GEOMETRIC x-x axis.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def I_yy(self):
        """
        The second moment of inertia about the GEOMETRIC y-y axis.
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
    def width(self):
        """
        The width of the section (max(x) - min(x)).
        """

        raise NotImplementedError

    @property
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

    def __init__(self, points: List[Point]):

        self.poly = Polygon(points)

    @property
    def area(self):
        return self.poly.area

    @property
    def I_xx(self):

        raise NotImplementedError

    @property
    def I_yy(self):

        raise NotImplementedError

    @property
    def principal_angle(self):

        raise NotImplementedError

    @property
    def J(self):

        raise NotImplementedError

    @property
    def centroid(self):

        return self.poly.centroid

    @property
    def depth(self):

        raise NotImplementedError

    @property
    def width(self):

        raise NotImplementedError

    @property
    def bounding_box(self) -> List[float]:

        return self.poly.bounds


class Rectangle(Section):
    def __init__(self, length, height):

        self.length = length
        self.height = height

    @property
    def area(self):
        return self.length * self.height

    @property
    def I_xx(self):

        return self.length * self.height ** 3 / 12

    @property
    def I_yy(self):

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

        return [
            Point(-self.length / 2, -self.height / 2),
            Point(self.length / 2, self.height / 2),
        ]


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
    def I_xx(self):

        I_xx = 0

        centroid = self.centroid

        for s, n in self.sections:

            I_xx += s.I_xx + s.area * (n.y - centroid.y) ** 2

        return I_xx

    @property
    def I_yy(self):

        I_yy = 0

        centroid = self.centroid

        for s, n in self.sections:

            I_yy += s.I_yy + s.area * (n.x - centroid.x) ** 2

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
