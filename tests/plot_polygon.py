"""
Test file for the build_patch method of the section_prop file.

Should show a 10x10 square with a 5x5 triangle inside it. The triangle should be hollow
"""

import matplotlib.pyplot as plt
import section_prop
from shapely.geometry import Polygon, polygon

fig, ax = plt.subplots()

ext = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
hole = [(2.5, 2.5), (7.5, 2.5), (7.5, 7.5)]

p = Polygon(ext, [hole])
p = polygon.orient(p)  # needed to properly orient the hole

patch = section_prop.build_patch(p)

ax.set_xlim(-2.5, 12.5)
ax.set_ylim(-2.5, 12.5)
ax.set_aspect(1.0)
ax.add_patch(patch)

fig.show()
