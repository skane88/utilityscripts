"""
This file defines a minimal set of units for use in engineering calculations.

It relies on the Pint units package.

Also imports math & sets up pi
"""

import unyt
from math import pi

# define a unicode pi value for pretty printing.
π = pi

# units of length
m = unyt.metre
inch = unyt.inch
ft = unyt.foot
yd = unyt.yard
mi = unyt.mile

# test comment

# derived units of length
mm = unyt.mm
cm = unyt.cm
km = unyt.km
unyt.define_unit("naut_mi", 1852 * m)
naut_mi = unyt.naut_mi

# units of time
s = unyt.second
minutes = unyt.minute
days = unyt.day
hr = unyt.hour
yr = unyt.year

unyt.define_unit("week", 7 * days)
week = unyt.week

# units of frequency
Hz = unyt.Hz

# units of mass
g = unyt.gram
kg = unyt.kg
t_us = unyt.ton
lb = unyt.pound

unyt.define_unit("t", 1000 * kg, prefixable=True)
t = unyt.t

# units of energy
J = unyt.J

kJ = unyt.kJ
MJ = unyt.MJ

# units of power
W = unyt.W

kW = unyt.kW
MW = unyt.MW

# units of current
A = unyt.A
mA = unyt.mA

# units of voltage
V = unyt.V

# units of force
N = unyt.newton
lb_f = unyt.pound_force

kN = unyt.kN
MN = unyt.MN

kip = unyt.kip

# units of pressure
Pa = unyt.pascal
psi = unyt.psi
bar = unyt.bar

kPa = unyt.kPa
MPa = unyt.MPa
GPa = unyt.GPa

ksi = unyt.ksi

unyt.define_unit("psf", 1 * lb_f / (ft ** 2))
psf = unyt.psf

# units of angle
deg = unyt.degree
rad = unyt.radian

# units of temperature
degC = unyt.degC
degK = unyt.K
degF = unyt.degF

# units of area
unyt.define_unit("ha", (100 * m) ** 2)
ha = unyt.ha

# units of volume
unyt.define_unit(symbol="L", value=0.001 * (m ** 3), prefixable=True)
L = unyt.L

# units of velocity
unyt.define_unit(symbol="knot", value=1852 * m / hr)
knot = unyt.knot

# constants
g_acc = unyt.standard_gravity

# set default printing
unyt.default_format = ".3f~P"
