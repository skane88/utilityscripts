"""
Utilities for working with loads
"""

from math import pi


def pipe_load(*, pipe_density, content_density, outer_diameter, inner_diameter):
    """
    Calculate loads in a pipe.

    :param pipe_density: The density of the pipe.
    :param content_density: The density of the contents.
    :param outer_diameter: The outer diameter.
    :param inner_diameter: The inner diameter.
    :return: A dictionary of masses.
    """

    total_area = 0.25 * pi * outer_diameter**2
    inner_area = 0.25 * pi * inner_diameter**2

    pipe_area = total_area - inner_area

    empty_weight = pipe_area * pipe_density
    content_weight = inner_area * content_density
    full_weight = empty_weight + content_weight

    return {
        "empty_weight": empty_weight,
        "content_weight": content_weight,
        "full_weight": full_weight,
    }
