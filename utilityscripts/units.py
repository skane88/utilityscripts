"""
This file defines a minimal set of units for use in engineering calculations.

It relies on the Pint units package.

Also imports math & sets up pi
"""

from math import acos, asin, atan, cos, degrees, pi, radians, sin, tan  # noqa: F401

import pint
import pint_pandas  # noqa: F401

from utilityscripts.steel import c_sections_df, i_sections_df

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

I_SECTIONS_DF_UNITS = i_sections_df().astype(
    {
        "mass": "pint[kg/m]",
        "d": "pint[m]",
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
    }
)
C_SECTIONS_DF_UNITS = c_sections_df().astype(
    {
        "mass": "pint[kg/m]",
        "d": "pint[m]",
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
    }
)
