"""
Earthquake loads to AS1170.4
"""

from enum import StrEnum


class SoilType(StrEnum):
    """
    The type of soil
    """

    Ae = "Ae"
    Be = "Be"
    Ce = "Ce"
    De = "De"
    Ee = "Ee"
