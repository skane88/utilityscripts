"""
File to contain some utilities for working with concrete.
"""

import re
from math import pi

MESH_DATA = {
    "RL1218": {"bar_dia": 11.9, "pitch": 100, "cross_bar_dia": 7.6, "cross_pitch": 200},
    "RL1018": {"bar_dia": 9.5, "pitch": 100, "cross_bar_dia": 7.6, "cross_pitch": 200},
    "RL818": {"bar_dia": 7.6, "pitch": 100, "cross_bar_dia": 7.6, "cross_pitch": 200},
    "SL102": {"bar_dia": 9.5, "pitch": 200, "cross_bar_dia": 9.5, "cross_pitch": 200},
    "SL92": {"bar_dia": 8.6, "pitch": 200, "cross_bar_dia": 8.6, "cross_pitch": 200},
    "SL82": {"bar_dia": 7.6, "pitch": 200, "cross_bar_dia": 7.6, "cross_pitch": 200},
    "SL72": {"bar_dia": 6.75, "pitch": 200, "cross_bar_dia": 6.75, "cross_pitch": 200},
    "SL62": {"bar_dia": 6.0, "pitch": 200, "cross_bar_dia": 6.0, "cross_pitch": 200},
    "SL81": {"bar_dia": 7.6, "pitch": 100, "cross_bar_dia": 7.6, "cross_pitch": 200},
}


def circle_area(dia):

    return pi * (dia**2) / 4


def reo_area(
    bar_spec: str = None, width: float = 1000, main_direction: bool = True
) -> float:
    """
    Calculate areas of reinforcement from a standard Australian specification code.
    """

    bar_pattern = "([LN][0-9]+){1}"

    no_bars = re.compile(f"(([0-9]+)(-))*{bar_pattern}")
    bars_with_spacing = re.compile(f"{bar_pattern}(([-@]){{1}}([0-9]+)){{1}}")
    mesh = re.compile("(([SR]L){1}([0-9]+){1}){1}")

    is_no_bars = no_bars.fullmatch(bar_spec)
    is_bars_spacing = bars_with_spacing.fullmatch(bar_spec)
    is_mesh = mesh.fullmatch(bar_spec)

    all_matches = [is_no_bars, is_bars_spacing, mesh]

    if all(x is not None for x in all_matches):
        raise ValueError(
            "Expected bar specification to match only one regular expression."
        )

    if is_no_bars:

        no_bars = is_no_bars[2]

        no_bars = 1 if no_bars is None else int(no_bars)
        bar_type = is_no_bars[4]
        bar_dia = int(bar_type[1:])

    if is_bars_spacing:

        bar_type = is_bars_spacing[1]
        bar_dia = int(bar_type[1:])
        bar_spacing = is_bars_spacing[4]
        no_bars = width / int(bar_spacing)

    if is_mesh:

        mesh_data = MESH_DATA[bar_spec]

        bar_dia = mesh_data["bar_dia"]
        cross_bar_dia = mesh_data["cross_bar_dia"]
        pitch = mesh_data["pitch"]
        cross_pitch = mesh_data["cross_pitch"]

        if main_direction:
            bar_spacing = pitch
        else:
            bar_dia = cross_bar_dia
            bar_spacing = cross_pitch

        no_bars = width / int(bar_spacing)

    bar_area = 0.25 * pi * bar_dia**2

    return bar_area * no_bars
