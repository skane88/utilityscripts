"""
This file defines a minimal set of units for use in engineering calculations.

It relies on the Pint units package.

Also imports math & sets up pi
"""

import copy
from math import (
    acos,  # noqa: F401
    asin,  # noqa: F401
    atan,  # noqa: F401
    cos,  # noqa: F401
    degrees,  # noqa: F401
    pi,
    radians,  # noqa: F401
    sin,  # noqa: F401
    tan,  # noqa: F401
)

import pint
import pint_pandas  # noqa: F401

from utilityscripts.math_utils import m_ceil, m_floor, m_round
from utilityscripts.steel import (
    SteelGrade,
    c_section_df,
    i_section_df,
    nearest_standard_plate,  # noqa: F401
    nearest_standard_weld,  # noqa: F401
    standard_grades,  # noqa: F401
    standard_plate_df,
    standard_weld_df,
    steel_grade_df,
)

# define a unicode pi value for pretty printing if req'd.
π = pi

ureg = pint.UnitRegistry()
pint.set_application_registry(ureg)

# units of length
m = ureg.metre
um = ureg.micrometre
μm = ureg.micrometre
inch = ureg.inch
ft = ureg.foot
yd = ureg.yard
mi = ureg.mile

# derived units of length
mm = ureg.mm
cm = ureg.cm
km = ureg.km
ureg.define("naut_mi = 1852 * m")
naut_mi = ureg.naut_mi

# units of time
s = ureg.second
minute = ureg.minute
minutes = ureg.minute
day = ureg.day
days = ureg.day
hr = ureg.hour
hrs = ureg.hour
yr = ureg.year
yrs = ureg.year
week = ureg.week
weeks = ureg.week

# units of frequency
Hz = ureg.Hz
rpm = 1 / minute

# units of mass
g = ureg.gram
kg = ureg.kg
ton = ureg.t
lb = ureg.pound
ton_us = ureg.ton

# units of energy
J = ureg.J

kJ = ureg.kJ
MJ = ureg.MJ

# units of power
W = ureg.W

kW = ureg.kW
MW = ureg.MW

# units of current
A = ureg.A
mA = ureg.mA

# units of voltage
V = ureg.V

# units of force
N = ureg.newton
lb_f = ureg.pound_force

kN = ureg.kN
MN = ureg.MN

kip = ureg.kip

# units of moment
Nm = N * m
kNm = kN * m  # convenience definition
MNm = MN * m  # convenience definition

# units of pressure
Pa = ureg.pascal
psi = ureg.psi
bar = ureg.bar
psf = lb_f / (ft**2)

kPa = ureg.kPa
MPa = ureg.MPa
GPa = ureg.GPa

ksi = ureg.ksi

# units of angle
deg = ureg.degree
rad = ureg.radian

# units of temperature
degC = ureg.degC
degK = ureg.K
degF = ureg.degF

# units of area
ha = ureg.ha

# units of volume
L = ureg.L
floz = ureg.fluid_ounce

# units of velocity
knot = ureg.knot

# constants
g_acc = ureg.standard_gravity
gacc = g_acc  # convenience definition
rho_air = 1.2 * kg / m**3  # see AS1170.2
rho_water = 1000 * kg / m**3
rho_seawater = 1025 * kg / m**3
rho_concrete = 2400 * kg / m**3
rho_steel = 7850 * kg / m**3  # see AS4100


# set default printing
ureg.default_format = ".3f~P"


def steel_grade_df_units():
    """
    Creates a dataframe of standard Australian steel grades, with Pint units.
    """

    grade_df = steel_grade_df()

    return grade_df.astype({"t": "pint[m]", "f_y": "pint[MPa]", "f_u": "pint[MPa]"})


def i_section_df_units(grade: None | SteelGrade | dict[str, SteelGrade] = None):
    """
    Creates a dataframe of standard Australian I sections, with Pint units.

    :param grade: An optional SteelGrade object or dictionary to assign
        to the sections. For different section types (e.g. WB vs UB),
        specify the grade as a dictionary: {designation: SteelGrade}.
        If a designation is missed, sections will be assigned a grade
        of None.
        NOTE: the grade should not already have units on it.
    """

    section_df = i_section_df(grade=grade)

    return section_df.astype(
        {
            "mass": "pint[kg/m]",
            "d": "pint[m]",
            "d_1": "pint[m]",
            "b_f": "pint[m]",
            "t_f": "pint[m]",
            "t_w": "pint[m]",
            "r_1": "pint[m]",
            "w_1": "pint[m]",
            "r1_or_w1": "pint[m]",
            "a_g": "pint[m**2]",
            "i_x": "pint[m**4]",
            "i_y": "pint[m**4]",
            "z_x": "pint[m**3]",
            "z_y": "pint[m**3]",
            "s_x": "pint[m**3]",
            "s_y": "pint[m**3]",
            "r_x": "pint[m]",
            "r_y": "pint[m]",
            "j": "pint[m**4]",
            "i_w": "pint[m**6]",
            "f_yf": "pint[MPa]",
            "f_yw": "pint[MPa]",
            "f_uf": "pint[MPa]",
            "f_uw": "pint[MPa]",
        }
    )


def c_section_df_units(grade: None | SteelGrade | dict[str, SteelGrade] = None):
    """
    Creates a dataframe of standard Australian C sections, with Pint units.

    :param grade: An optional SteelGrade object or dictionary to assign
        to the sections. For different section types (e.g. WB vs UB),
        specify the grade as a dictionary: {designation: SteelGrade}.
        If a designation is missed, sections will be assigned a grade
        of None.
        NOTE: the grade should not already have units on it.
    """

    section_df = c_section_df(grade=grade)

    return section_df.astype(
        {
            "mass": "pint[kg/m]",
            "d": "pint[m]",
            "d_1": "pint[m]",
            "b_f": "pint[m]",
            "t_f": "pint[m]",
            "t_w": "pint[m]",
            "r_1": "pint[m]",
            "a_g": "pint[m**2]",
            "i_x": "pint[m**4]",
            "i_y": "pint[m**4]",
            "z_x": "pint[m**3]",
            "z_yl": "pint[m**3]",
            "z_yr": "pint[m**3]",
            "s_x": "pint[m**3]",
            "s_y": "pint[m**3]",
            "r_x": "pint[m]",
            "r_y": "pint[m]",
            "j": "pint[m**4]",
            "i_w": "pint[m**6]",
            "f_yf": "pint[MPa]",
            "f_yw": "pint[MPa]",
            "f_uf": "pint[MPa]",
            "f_uw": "pint[MPa]",
        }
    )


def standard_plate_df_units():
    """
    Get a DataFrame of standard Australian plate thicknesses, with assigned units.
    """

    plate_df = standard_plate_df()

    return plate_df.astype({"thickness": "pint[m]"})


def standard_weld_df_units():
    """
    Get a DataFrame of standard Australian weld sizes, with assigned units.
    """

    weld_df = standard_weld_df()

    return weld_df.astype({"leg_size": "pint[m]"})


def _prep_round(x, base):
    """
    Helper method to prepare for the m_round, m_floor and m_ceil methods.

    :param x: The number to round.
    :param base: Round to multiples of?
    :return: A tuple containing:
        (x, base, units, si_units)
        where units is the units of x, or None, and si_units is the base units of x.
    """

    if not isinstance(x, pint.Quantity):
        raise ValueError(f"x must be an instance of pint.Quantity. {x=}")

    if (
        isinstance(x, pint.Quantity)
        and isinstance(base, pint.Quantity)
        and x.to_base_units().units != base.to_base_units().units
    ):
        raise ValueError(
            "Both x and base must be units of the same type (e.g. both length, or both mass)."
            + f"{x=}, {base=}"
        )

    units = None
    si_units = None

    if isinstance(x, pint.Quantity):
        x = copy.deepcopy(x)
        units = x.units
        x.ito_base_units()
        si_units = x.units
        x = x.magnitude

    if isinstance(base, pint.Quantity):
        base = copy.deepcopy(base)
        base.ito_base_units()
        base = base.magnitude
    else:
        base = base * units
        base.ito_base_units()
        base = base.magnitude

    return x, base, units, si_units


def m_round_units(x, base, floor_val: bool | None = None):
    """
    Custom rounding function that works with units,
    and can round to the nearest multiple.

    :param x: The number to round.
    :param base: Round to multiples of?
    """

    (
        x,
        base,
        units,
        si_units,
    ) = _prep_round(x, base)

    val = m_round(x, base)

    if units is not None:
        val = (val * si_units).to(units)

    return val


def m_floor_units(x, base):
    """
    Custom rounding function that works with units,
    and can round to the nearest multiple.

    :param x: The number to round.
    :param base: Round to multiples of?
    """

    (
        x,
        base,
        units,
        si_units,
    ) = _prep_round(x, base)

    val = m_floor(x, base)

    if units is not None:
        val = (val * si_units).to(units)

    return val


def m_ceil_units(x, base):
    """
    Custom rounding function that works with units,
    and can round to the nearest multiple.

    :param x: The number to round.
    :param base: Round to multiples of?
    """

    (
        x,
        base,
        units,
        si_units,
    ) = _prep_round(x, base)

    val = m_ceil(x, base)

    if units is not None:
        val = (val * si_units).to(units)

    return val
