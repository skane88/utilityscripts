"""
Contains tools for working with my blog.
"""

import sys


def make_cmap(colors, position=None, bit=False):
    """
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.

    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    
    Function from: http://schubert.atmos.colostate.edu/~cslocum/custom_cmap.html
    """
    import matplotlib as mpl
    import numpy as np

    bit_rgb = np.linspace(0, 1, 256)
    if position == None:
        position = np.linspace(0, 1, len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (
                bit_rgb[colors[i][0]],
                bit_rgb[colors[i][1]],
                bit_rgb[colors[i][2]],
            )
    cdict = {"red": [], "green": [], "blue": []}
    for pos, color in zip(position, colors):
        cdict["red"].append((pos, color[0], color[0]))
        cdict["green"].append((pos, color[1], color[1]))
        cdict["blue"].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap("my_colormap", cdict, 256)
    return cmap


def color_list(N, cmap):
    """
    This function returns a list of N colours based on a colour map.
    """
    ret_list = []

    for i in range(N):
        ret_list.append(cmap(i / (N - 1)))

    return ret_list

BLOG_CMAP = make_cmap([(65, 131, 196), (206, 25, 13)], bit=True)

