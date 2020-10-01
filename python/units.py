"""
This file defines a minimal set of units for use in engineering calculations.

It relies on the Pint units package.

Also imports math & sets up pi
"""

import unyt
import math

# define a unicode pi value for pretty printing.
π = math.pi

# units of length
m = unyt.metre
inch = unyt.inch
ft = unyt.foot
yd = unyt.yard
mi = unyt.mile

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

unyt.define_unit("week", 7 * days)
week = unyt.week
hr = unyt.hour
yr = unyt.year

# units of mass
g = unyt.gram
kg = unyt.kg

unyt.define_unit("t", 1000 * kg, prefixable=True)
t = unyt.t
t_us = unyt.ton
lb = unyt.pound

# derived units of mass
kg = unyt.kg

# units of energy
J = unyt.J

# derived units of energy
kJ = unyt.kJ
MJ = unyt.MJ

# units of force
N = unyt.newton
lb_f = unyt.pound_force

unyt.define_unit("kip", 1000 * lb_f)
kip = unyt.kip

# derived units of force
kN = unyt.kN
MN = unyt.MN

# units of pressure
Pa = unyt.pascal
psi = unyt.psi
bar = unyt.bar

# derived units of pressure
kPa = unyt.kPa
MPa = unyt.MPa
GPa = unyt.GPa

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
unyt.define_unit(symbol="l", value=0.001 * (m ** 3), prefixable=True)
l = unyt.l

# units of velocity
unyt.define_unit(symbol="knot", value=1852 * m / hr)
knot = unyt.knot

# constants
g_acc = unyt.standard_gravity

# set default printing
unyt.default_format = ".3f~P"
