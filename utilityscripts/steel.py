"""
To contain some utilities for steel design
"""
from dataclasses import dataclass
from math import asin, atan, cos, radians, sin

import matplotlib.pyplot as plt
import shapely as shp
from shapely.plotting import plot_polygon

from utilityscripts.section_prop import build_circle


def alpha_m(*, M_m, M_2, M_3, M_4):
    """
    Determines the moment modification factor as per AS4100 S5.6.1.1.a.iii

    :param M_m: The maximum moment.
    :param M_2: The moment at the 1st 1/4 point.
    :param M_3: The moment at midspan.
    :param M_4: The moment at the 2nd 1/4 point.
    """

    return 1.7 * M_m / (M_2**2 + M_3**2 + M_4**2) ** 0.5


def alpha_v(*, d_p, t_w, f_y, s, f_yref=250.0):
    """
    Calculate the stiffened web shear buckling parameter alpha_v as per
    AS4100 S5.11.5.2.

    :param d_p: The depth of the web panel.
    :param t_w: The thickness of the web.
    :param f_y: The yield strength of the web panel.
    :param s: The length of the web or spacing of vertical stiffeners that meet the
        requirement of AS4100.
    :param f_yref: The reference yield stress, nominally 250.
    """

    a1 = (82 / ((d_p / t_w) * (f_y / f_yref) ** 0.5)) ** 2

    a2_param = 1.0 if (s / d_p) <= 1.0 else 0.75

    a2 = (a2_param / ((s / d_p) ** 2)) + 1.0

    return min(a1 * a2, 1.0)


def alpha_d(*, alpha_v, d_p, s):
    """
    Calculate the stiffened web shear buckling parameter alpha_d as per
    AS4100 S5.11.5.2.

    Does not check for the presence of a stiffened end post.

    :param alpha_v: The stiffened web shear buckling parameter alpha_v as per
        AS4100 S5.11.5.2.
    :param d_p: The depth of the web panel.
    :param s: The length of the web or spacing of vertical stiffeners that meet the
        requirement of AS4100.
    """

    return 1 + ((1 - alpha_v) / (1.15 * alpha_v * (1 + (s / d_p) ** 2) ** 0.5))


def as_s5_15_3(*, gamma, a_w, alpha_v, v_star, v_u, d_p, s, phi=0.9):
    """
    Calculate the minimum area of a transverse shear stiffener as per AS4100 S5.15.3

    :param gamma: The stiffener type factor.
    :param a_w: The area of the web.
    :param alpha_v: The stiffened web shear buckling parameter alpha_v as per
        AS4100 S5.11.5.2.
    :param v_star: The shear in the web.
    :param v_u: The ultimate shear capacity of the web.
    :param d_p: The depth of the web panel.
    :param s: The length of the web or spacing of vertical stiffeners that meet the
        requirement of AS4100.
    :param phi: The capacity reduction factor
    """

    a_1 = 0.5 * gamma * a_w
    a_2 = 1 - alpha_v
    a_3 = v_star / (phi * v_u)
    a_4 = (s / d_p) - ((s / d_p) ** 2 / (1 + (s / d_p) ** 2) ** 0.5)

    return a_1 * a_2 * a_3 * a_4


def v_by(*, d_f, t_p, f_up):
    """
    Calculate the bolt hole bearing capacity, as limited by local yielding, as per
    AS4100 S9.2.2.4

    :param d_f: The fastener diameter.
    :param t_p: The thickness of the plate.
    :param f_up: The ultimate tensile stress of the plate.
    """

    return 3.2 * d_f * t_p * f_up


def v_bt(*, a_e, t_p, f_up):
    """
    Calculate the bolt hole bearing capacity, as limited by tearout, as per
    AS4100 S9.2.2.4

    :param a_e: Fastener edge distance.
    :param t_p: Thickness of plate.
    :param f_up: The ultimate tensile stress of the plate.
    """

    return a_e * t_p * f_up


@dataclass
class Lug:
    t: float  # thickness of the plate.
    b: float  # width of the lug
    h: float  # height to the centre of the hole
    r: float  # radius of the outside of the lug, centred on the hole
    dia_hole: float  # Diameter of the hole
    e_hole: float = 0.0  # eccentricity of the hole.

    @property
    def r_hole(self):
        return self.dia_hole / 2

    @property
    def x_cp(self):
        """
        The x co-ordinate of the centre of the hole, from the bottom LHS corner.
        """

        return self.b / 2 + self.e_hole

    @property
    def y_cp(self):
        """
        The y co-ordinate of the centre of the hole, from the bottom LHS corner.
        """
        return self.h

    @property
    def cp(self):
        """
        Return the centre of the circle as a tuple (x, y)
        """

        return self.x_cp, self.y_cp

    @property
    def _l_lhs_ctr(self):
        """
        Distance from the left hand corner to the centre of the lug.
        """

        return (self.x_cp**2 + self.y_cp**2) ** 0.5

    @property
    def _theta1_lhs_ctr(self):
        """
        Angle from the LHS base of the lug to the centre of the lug.
        """

        return asin(self.y_cp / self._l_lhs_ctr)

    @property
    def _theta2_lhs_tp1(self):
        """
        The angle between the lines formed by the LHS corner to the tangent point
        and the LHS corner to the centre of the lug.
        """

        return asin(self.r / self._l_lhs_ctr)

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

        return (self._l_lhs_ctr**2 - self.r**2) ** 0.5

    @property
    def x_tp1(self):
        """
        The x co-ordinate of the LHS tangent point, tp1.
        """

        return self._l_lhs_tp1 * cos(self._theta_lhs_tp1)

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

        return ((self.b - self.x_cp) ** 2 + self.y_cp**2) ** 0.5

    @property
    def _theta1_rhs_ctr(self):
        """
        Angle from the base of the lug at the RHS to the centre of the lug.
        """
        return asin(self.y_cp / self._l_rhs_ctr)

    @property
    def _theta2_rhs_tp2(self):
        """
        The angle between the lines formed by the RHS corner to the tangent point
        and the LHS corner to the centre of the lug.
        """

        return asin(self.r / self._l_rhs_ctr)

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

        return (self._l_rhs_ctr**2 - self.r**2) ** 0.5

    @property
    def x_tp2(self):
        """
        The x co-ordinate of the RHS tangent point, tp2.
        """

        return self.b - (self._l_rhs_tp2 * cos(self._theta_rhs_tp2))

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

        y_tp1_ctr = self.y_tp1 - self.y_cp
        x_tp1_ctr = self.x_cp - self.x_tp1

        return atan(y_tp1_ctr / x_tp1_ctr)

    @property
    def _theta3_tp2(self):
        """
        The horizontal angle from the centre of the circle to the tangent point.
        """

        y_tp2_ctr = self.y_tp2 - self.y_cp
        x_tp2_ctr = self.x_tp2 - self.x_cp

        return atan(y_tp2_ctr / x_tp2_ctr)

    def lug_polygon(self, no_points=64):
        """
        Return a Shapely polygon describing the lug.

        :param no_points: the no. of points to use for the circular sections of the lug.
        """

        points = [(0, 0), (self.b, 0)]

        points += build_circle(
            centroid=self.cp,
            radius=self.r,
            no_points=no_points,
            limit_angles=(self._theta3_tp2, radians(180) - self._theta3_tp1),
        )

        points.append((0, 0))

        hole_points = build_circle(
            centroid=(self.x_cp, self.y_cp),
            radius=self.dia_hole / 2,
            no_points=no_points,
        )
        hole_points.reverse()

        lug = shp.Polygon(shell=points, holes=[hole_points])

        return lug

    def plot_lug(
        self,
        ax=None,
        face_color=(0.85, 0.85, 0.85, 1.0),
        edge_color=(0, 0, 0),
        add_points=False,
    ):
        """
        Plot the lug. Uses Shapely's plotting library to plot the lug.

        Also returns a tuple of:

        if add_points is False:
            (PathPatch, None)
        else:
            (PathPatch, Line2D)

        :param ax: The matplotlib axis on which to plot the lug.
        :param face_color: A matplotlib color specification to color the face.
        :param edge_color: A matplotlib color specification for the edge.
        :param add_points: Plot the vertices of the lug?
        """

        lug_patch = plot_polygon(
            self.lug_polygon(),
            add_points=add_points,
            facecolor=face_color,
            edgecolor=edge_color,
        )

        plt.show()

        if add_points is False:
            return (lug_patch, None)
        else:
            return lug_patch
