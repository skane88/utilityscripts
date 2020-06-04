"""
Calculates basic section properties
"""

import math
from typing import List, Tuple
import abc


class Node:
    """
    Node class in 2D
    """

    def __init__(self, x, y):

        self.x = x
        self.y = y

    def __add__(self, other):

        if not isinstance(other, Node):
            raise NotImplementedError

        return Node(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other):

        if not isinstance(other, Node):
            raise NotImplementedError

        return Node(x=self.x - other.x, y=self.y - other.y)

    def __repr__(self):

        return f"{type(self).__name__}({self.x}, {self.y})"


class Section(abc.ABCMeta):
    """
    Parent section class
    """

    def __init__(self, sect_name=None):

        self.sect_name = sect_name

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
    @abc.abstractmethod
    def width(self):
        """
        The width of the section (max(x) - min(x)).
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def bounding_box(self) -> List[Node]:
        """
        The bounding box of the section. Consists of 2x Node objects in a list:

            [Node(x=min(x), y=min(y)), Node(x=max(x), y=max(y))]
        """

        raise NotImplementedError


class Rectangle(Section):
    def __init__(self, length, height, sect_name=None):

        super().__init__(sect_name=sect_name)

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

        return Node(x=self.length / 2, y=self.height / 2)

    @property
    def depth(self):

        return self.height

    @property
    def width(self):

        return self.length

    @property
    def bounding_box(self) -> List[Node]:

        return [
            Node(-self.length / 2, -self.height / 2),
            Node(self.length / 2, self.height / 2),
        ]

    @classmethod
    def create_square(cls, side, sect_name=None):

        return cls(length=side, height=side, sect_name=sect_name)


class CombinedSection(Section):
    def __init__(self, sections: List[Tuple[Section, Node]], sect_name=None):
        """

        :param sections: A list of sections & centroids
        """

        super().__init__(sect_name=sect_name)

        all_sections = []
        for s, n in sections:

            if isinstance(s, CombinedSection):

                for t, o in s.sections:

                    all_sections.append(t, n + o)

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

        return Node(x=mx / self.area, y=my / self.area)

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
    def bounding_box(self) -> List[Node]:

        test_bounding_box = None

        for s, n in self.sections:

            if test_bounding_box is None:

                test_bounding_box = [s.bounding_box[0] + n, s.bounding_box[1] + n]

            else:

                sect_min = s.bounding_box[0] + n
                sect_max = s.bounding_box[1] + n

                combined_min = test_bounding_box[0]
                combined_max = test_bounding_box[1]

                max_x = max(sect_max.x, combined_max.x)
                max_y = max(sect_max.y, combined_max.y)

                min_x = min(sect_min.x, combined_min.x)
                min_y = min(sect_min.y, combined_min.y)

                test_bounding_box = [Node(x=min_x, y=min_y), Node(x=max_x, y=max_y)]

        return test_bounding_box

    def add_element(self, section, centroid):

        self.sections.append((section, centroid))

    def move_to_centre(self):

        sections = []
        centroid = self.centroid

        for s, n in self.sections:

            sections.append((s, n - centroid))

        return CombinedSection(sections=sections, sect_name=self.sect_name)

    @classmethod
    def make_I(cls, b_f, d, t_f, t_w, sect_name=None):

        d_w = d - 2 * t_f

        top_flange = Rectangle(length=b_f, height=t_f)
        bottom_flange = Rectangle(length=b_f, height=t_f)
        web = Rectangle(length=t_w, height=d_w)

        depth = t_f * 2 + d_w

        n_tf = Node(x=b_f / 2, y=depth - t_f / 2)
        n_w = Node(x=b_f / 2, y=t_f + d_w / 2)
        n_bf = Node(x=b_f / 2, y=t_f / 2)

        return cls(
            sections=[(top_flange, n_tf), (bottom_flange, n_bf), (web, n_w)],
            sect_name=sect_name,
        ).move_to_centre()
