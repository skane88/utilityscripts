"""
Utilities for working with loads
"""

from enum import StrEnum
from math import pi


class RoofType(StrEnum):
    ACCESSIBLE = "accessible"
    ACCESSIBLEFROMGROUND = "accessible from ground"
    OTHER = "other"


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


def roof_load_as1170_1(
    roof_type: RoofType | str, area: float | None
) -> tuple[float, float]:
    """
    Calculate roof load as per AS1170.1 S3.5

    Notes
    -----
    Ignores the distinction between structure and cladding
    that AS1170 allows for OTHER type roofs.

    Parameters
    ----------
    roof_type : RoofType | str
        What sort of roof is it?
    area : float | None
        The tributary area.

    Returns
    -------
    tuple[float, float]

    The loads to apply as (pressure in kPa, point load in kN)
    """

    if roof_type == RoofType.ACCESSIBLE:
        return 1.5, 1.8

    if roof_type == RoofType.ACCESSIBLEFROMGROUND:
        return 1.0, 1.8

    if roof_type != RoofType.OTHER:
        raise ValueError(
            "Roof type should be a valid RoofType Enum or equivalent string."
        )

    return max(0.25, 1.8 / area + 0.12), 1.4
