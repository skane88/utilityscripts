"""
This file defines a minimal set of units for use in engineering calculations.

It relies on the Pint units package.

Also imports math & sets up pi
"""

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

from utilityscripts.math_utils import m_ceil, m_floor, m_round

# define a unicode pi value for pretty printing if req'd.
π = pi

ureg = pint.UnitRegistry()
pint.set_application_registry(ureg)

# gravity
g_acc = ureg.standard_gravity
gacc = g_acc  # convenience definition

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

# weight forces
lb_f = ureg.pound_force
kg_f = kg * gacc
ton_f = ton * gacc

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
rho_air = 1.2 * kg / m**3  # see AS1170.2
rho_water = 1000 * kg / m**3
rho_seawater = 1025 * kg / m**3
rho_concrete = 2400 * kg / m**3
rho_steel = 7850 * kg / m**3  # see AS4100


# set default printing
ureg.formatter.default_format = ".3f~P"


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
            "Both x and base must be units of the same type "
            + "(e.g. both length, or both mass)."
            + f"{x=}, {base=}"
        )

    units = None
    si_units = None

    if isinstance(x, pint.Quantity):
        units = x.units
        x = x.to_base_units()
        si_units = x.units
        x = x.magnitude

    if isinstance(base, pint.Quantity):
        base = base.to_base_units().magnitude
    else:
        base = (base * units).to_base_units().magnitude

    return x, base, units, si_units


def m_round_units(x: pint.Quantity, base: pint.Quantity | float):
    """
    Custom rounding function that works with units,
    and can round to the nearest multiple.
    Number is returned in the units of x.

    :param x: The number to round.
    :param base: Round to multiples of?
        If unitless (float, int etc.) then only the magnitude of x will be rounded.
        If a Quantity, then x will be rounded in correct units, but will be
        converted back to the original units before the result is returned.
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


def m_floor_units(x: pint.Quantity, base: pint.Quantity | float):
    """
    Custom rounding function that works with units,
    and can round to the nearest multiple.
    Number is returned in the units of x.

    :param x: The number to round.
    :param base: Round to multiples of?
        If unitless (float, int etc.) then only the magnitude of x will be rounded.
        If a Quantity, then x will be rounded in correct units, but will be
        converted back to the original units before the result is returned.
    """

    # perform a preliminary round of x to 1/10 of base, to get rid of potential
    # floating point errors.
    x = m_round_units(x, base / 100)

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


def m_ceil_units(x: pint.Quantity, base: pint.Quantity | float):
    """
    Custom rounding function that works with units,
    and can round to the nearest multiple.
    Number is returned in the units of x.

    :param x: The number to round.
    :param base: Round to multiples of?
        If unitless (float, int etc.) then only the magnitude of x will be rounded.
        If a Quantity, then x will be rounded in correct units, but will be
        converted back to the original units before the result is returned.
    """

    x = m_round_units(x, base / 100)

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
