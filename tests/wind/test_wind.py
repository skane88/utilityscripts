"""
Tests for the wind module.
"""

from utilityscripts.wind.as1170_2 import WindRegion, WindSite
from utilityscripts.wind.wind import OpenStructure


def test_open_structure():
    height = 10.0
    length = 10.0
    spacing = 10.0
    wind_site = WindSite(
        wind_region=WindRegion.C,
        terrain_category=2.0,
    )

    structure = OpenStructure(
        frame_h=height,
        frame_l=length,
        frame_s=spacing,
        wind_site=wind_site,
    )

    assert structure
    assert structure.frame_h == height
    assert structure.frame_l == length
    assert structure.frame_s == spacing
    assert structure.wind_site == wind_site
    assert structure.member_data.is_empty()

    structure = structure.add_member(
        name="member1",
        depth=1.0,
        length=1.0,
        reference_height=1.0,
        drag_coefficient=1.0,
        no_per_frame=1,
        no_unshielded_frames=1,
        no_shielded_frames=1,
    )

    assert not structure.member_data.is_empty()
