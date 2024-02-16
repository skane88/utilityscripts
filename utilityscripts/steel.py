"""
To contain some utilities for steel design
"""
from math import pi

from sectionproperties.analysis.section import Section
from sectionproperties.pre.geometry import Geometry
from sectionproperties.pre.library.primitive_sections import (
    rectangular_section,
    triangular_section,
)
from sectionproperties.pre.library.steel_sections import mono_i_section


def alpha_m(*, M_m, M_2, M_3, M_4):
    """
    Determines the moment modification factor as per AS4100 S5.6.1.1.a.iii

    :param M_m: The maximum moment.
    :param M_2: The moment at the 1st 1/4 point.
    :param M_3: The moment at mid-span.
    :param M_4: The moment at the 2nd 1/4 point.
    """

    return 1.7 * M_m / (M_2**2 + M_3**2 + M_4**2) ** 0.5


def alpha_v(*, d_p, t_w, f_y, s, f_y_ref=250.0):
    """
    Calculate the stiffened web shear buckling parameter alpha_v as per
    AS4100 S5.11.5.2.

    :param d_p: The depth of the web panel.
    :param t_w: The thickness of the web.
    :param f_y: The yield strength of the web panel.
    :param s: The length of the web or spacing of vertical stiffeners that meet the
        requirement of AS4100.
    :param f_y_ref: The reference yield stress, nominally 250.
    """

    a1 = (82 / ((d_p / t_w) * (f_y / f_y_ref) ** 0.5)) ** 2

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
    Calculate the bolt hole bearing capacity, as limited by tear-out, as per
    AS4100 S9.2.2.4

    :param a_e: Fastener edge distance.
    :param t_p: Thickness of plate.
    :param f_up: The ultimate tensile stress of the plate.
    """

    return a_e * t_p * f_up


def jack_bolt_effort(*, load, p, r_effort, r_bolt, mu, against: bool = True):
    """
    Calculate the force required to turn a jacking bolt.

    :param load: The weight or load.
    :param p: The pitch of the screw
    :param r_effort: The radius of the effort force.
    :param r_bolt: The radius of the screw / bolt.
    :param mu: The assumed coefficient of friction.
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
    *, point_force, f_y, phi=0.9, width_lever: tuple[float, float] = None
):
    """
    Calculate the local thickness of a plate required to transfer a load back to a support by local bending.
    This assumes a 45deg load dispersion through the plate towards the support, ignoring and load transfer into
    the 2nd dimension.

    :param point_force: The load to transfer.
    :param f_y: The yield stress of the plate.
    :param phi: A capacity reduction factor.
    :param width_lever: A tuple containing the width of the applied load and the lever arm between load and support.
        If None, the load is treated as a point load.
    """

    if width_lever is None:
        return ((2 * point_force) / (phi * f_y)) ** 0.5

    lw = width_lever[0]
    lever = width_lever[1]

    if lever == 0:
        width_lever_ratio = 0
    else:
        width_lever_ratio = lw / lever

    return 2 / (((phi * f_y / point_force) * (width_lever_ratio + 2)) ** 0.5)


def make_i_section(
    *, b_f, d, t_f, t_w, b_fb=None, t_fb=None, corner_radius=None, n_r=8, weld_size=None
) -> Geometry:
    """
    Make the geometry for an I section.

    A thin wrapper around section-properties' mono_i_section.

    :param b_f: The width of the top flange.
    :param d: The section depth.
    :param t_f: The top flange thickness.
    :param t_w: The web thickness.
    :param b_fb: The width of the bottom section.
        Provide None for a symmetric section.
    :param t_fb: The bottom flange thickness.
        Provide None if both flanges the same.
    :param corner_radius: The corner radius of the i-section.
        NOTE: If both a corner radius and a weld size are defined,
        priority is given to corner radii.
        If a weld is present, set corner_radius = None.
    :param n_r: The no. of points used to model the corner.
    :param weld_size: The corner fillet weld (if any).
        NOTE: If both a corner radius and a weld size are defined,
        priority is given to corner radii.
        If a weld is present, set corner_radius = None.
    """

    if b_fb is None:
        b_fb = b_f

    if t_fb is None:
        t_fb = t_f

    i_sect = mono_i_section(
        b_t=b_f,
        b_b=b_fb,
        d=d,
        t_ft=t_f,
        t_fb=t_fb,
        t_w=t_w,
        r=corner_radius if corner_radius is not None else 0,
        n_r=n_r,
    ).align_center(align_to=(0, 0))

    if corner_radius is None and weld_size is not None:
        i_sect = i_sect.align_to(other=(0, 0), on="top")

        weld_br = triangular_section(b=weld_size, h=weld_size)
        weld_bl = weld_br.mirror_section(axis="y", mirror_point=(0, 0))
        weld_tr = weld_br.mirror_section(axis="x", mirror_point=(0, 0))
        weld_tl = weld_bl.mirror_section(axis="x", mirror_point=(0, 0))

        i_sect = i_sect + weld_br.shift_section(x_offset=t_w / 2, y_offset=t_fb)
        i_sect = i_sect + weld_bl.shift_section(x_offset=-t_w / 2, y_offset=t_fb)
        i_sect = i_sect + weld_tr.shift_section(x_offset=t_w / 2, y_offset=d - t_f)
        i_sect = i_sect + weld_tl.shift_section(x_offset=-t_w / 2, y_offset=d - t_f)

        i_sect.align_center(align_to=(0, 0))

    return i_sect


def make_section(
    geometry: Geometry, alpha_mesh_size=200, calculate_properties: bool = True
) -> Section:
    """
    Turn a sectionproperties Geometry object into a Section object.

    :param geometry: The Geometry object to turn into the Section object.
    :param alpha_mesh_size: The area of the largest mesh element,
        as a fraction of the overall area of the section.
    :param calculate_properties: Calculate the properties?
    """

    area = geometry.calculate_area()
    sect = Section(geometry.create_mesh(mesh_sizes=area / alpha_mesh_size))

    if calculate_properties:
        sect.calculate_geometric_properties()
        sect.calculate_warping_properties()
        sect.calculate_plastic_properties()

    return sect


class Overplated_Section:
    """
    Creates an overplated section with an overplate on the top & bottom.

    NOTE: Overplates are centred on the section.
    NOTE 2: This will work best with a symmetric or mono-symmetric I section,
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

        :param base_section: The section to overplate.
        :param t_op_top: The thickness of the top overplate.
            Use None if no overplate on top.
        :param b_op_top: The width of the top overplate.
            Use None if no overplate on top.
        :param t_op_bottom: The thickness of the bottom overplate.
            Use None if no overplate on bottom.
        :param b_op_bottom: The width of the bottom overplate.
            Use None if no overplate on the bottom.
        :param alpha_mesh_size: How big should the largest mesh element be,
            as a fraction of the total area of the shape?
        """

        self.base_section = base_section
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
