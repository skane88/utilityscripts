"""
File to contain some utilities for working with concrete.
"""

from math import pi

from geometry import Circle


def circle_area(dia):

    return pi * (dia**2) / 4


def reo_area(bar_spec: str = None):
    """
    Calculate areas of reinforcement from a standard Australian specification code.
    """

    pass
