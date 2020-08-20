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

# derived units of length
mm = unyt.mm
km = unyt.km
naut_mi = 1852 * m

# units of time
s = unyt.second
minutes = unyt.minute
days = unyt.day
weeks = 7 * days
hr = unyt.hour
yr = unyt.year

# units of mass
g = unyt.gram
kg = unyt.kg
t = 1000 * kg
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
kip = 1000 * lb_f

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
ha = (100 * m) ** 2

# units of volume
l = 0.001 * m ** 3
ml = l / 1000

# units of velocity
knot = 1852 * m / hr

# constants
g_acc = unyt.standard_gravity

# set default printing
unyt.default_format = ".3f~P"
