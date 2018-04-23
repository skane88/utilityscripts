"""
This file defines a minimal set of units for use in engineering calculations.

It relies on the Pint units package

Also imports math & sets up pi
"""

import pint
from math import pi

# define a unicode pi value for pretty printing.
π = pi

# set up a unit registry
units = pint.UnitRegistry()

# units of length
m = units.metre
inch = units.inch
ft = units.foot
yd = units.yard
mi = units.mile

# derived units of length
mm = units.mm
km = units.km
knot = units.knot

# units of time
s = units.second
minutes = units.minute
hr = units.hour
yr = units.year

# units of mass
g = units.gram
t = units.tonne
t_us = units.ton
lb = units.pound
t_us = units.ton

# derived units of mass
kg = units.kg

# units of energy
J = units.J

# derived units of energy
kJ = units.kJ

# units of force
N = units.newton
lb_f = units.pound_force
kip = units.kip

# derived units of force
kN = units.kN
MN = units.MN

# units of pressure
Pa = units.pascal
psi = units.psi

# derived units of pressure
kPa = units.kPa
MPa = units.MPa
GPa = units.GPa

# units of angle
deg = units.degree
rad = units.radian

# units of temperature
degC = units.degC
degK = units.degK
ΔC = units.delta_degC

# constants
g_acc = 9.80665 * m / s**2

# set default printing
units.default_format = '.3f~P'