"""
Tests for the wind module.
"""

from math import isclose

import polars as pl

from utilityscripts.wind.as1170_2 import SectionType
from utilityscripts.wind.wind import OpenStructure


def test_open_structure():
    height = 10.0
    length = 10.0
    spacing = 10.0
    terrain_category = 2.0

    structure = OpenStructure(
        frame_h=height,
        frame_l=length,
        frame_s=spacing,
        v_r=66.0,
        terrain_category=terrain_category,
    )

    assert structure
    assert structure.frame_h == height
    assert structure.frame_l == length
    assert structure.frame_s == spacing
    assert structure.terrain_category == terrain_category
    assert structure.member_data.is_empty()

    structure = structure.add_member(
        component_id="001",
        name="member1",
        depth=1.0,
        length=1.0,
        reference_height=1.0,
        drag_coefficient=1.0,
        no_per_frame=1,
        no_unshielded_frames=1,
        no_shielded_frames=1,
        include_in_solidity=True,
        circular_or_sharp=SectionType.CIRCULAR,
        master_component="comp1",
        comments="test",
    )

    assert not structure.member_data.is_empty()
    assert len(structure.member_data) == 1

    structure = structure.add_members(
        {
            "component_id": ["002"],
            "name": ["member2"],
            "master_component": ["comp1"],
            "comments": ["test"],
            "depth": [1.0],
            "length": [1.0],
            "reference_height": [1.0],
            "drag_coefficient": [1.0],
            "no_per_frame": [1],
            "no_unshielded_frames": [1],
            "no_shielded_frames": [1],
            "include_in_solidity": [True],
            "circular_or_sharp": [SectionType.CIRCULAR],
            "inclination": [90.0],
            "k_sh": [0.20],
        }
    )

    assert not structure.member_data.is_empty()
    assert len(structure.member_data) == 2  # noqa: PLR2004


def test_open_structure_example():
    height = 2.286
    length = 14.335
    spacing = 1.400
    terrain_category = 2.0

    structure = OpenStructure(
        frame_h=height,
        frame_l=length,
        frame_s=spacing,
        v_r=69.442,
        terrain_category=terrain_category,
        m_d=0.90,
    )

    structure = structure.add_members(
        {
            "component_id": ["0001", "0002", "0003", "0004"],
            "name": [
                "125x10EA Top Chord",
                "125x10EA Bottom Chord",
                "65x6EA Verticals",
                "125PFC Verticals",
            ],
            "master_component": ["Gantry", "Gantry", "Gantry", "Gantry"],
            "comments": ["test", "test", "test", "test"],
            "depth": [0.125, 0.125, 0.065, 0.065],
            "length": [14.335, 14.335, 2.286, 2.286],
            "reference_height": [23.104, 20.818, 21.961, 21.961],
            "drag_coefficient": [2.00, 2.00, 2.00, 0.60],
            "no_per_frame": [1, 1, 4, 4],
            "no_unshielded_frames": [1, 1, 1, 1],
            "no_shielded_frames": [1, 1, 1, 1],
            "include_in_solidity": [True, True, True, True],
            "circular_or_sharp": [
                SectionType.SHARP,
                SectionType.SHARP,
                SectionType.SHARP,
                SectionType.SHARP,
            ],
            "inclination": [90.0, 90.0, 90.0, 90.0],
            "k_sh": [0.20, 0.20, 0.20, 0.20],
        }
    )

    assert len(structure.member_data) == 4  # noqa: PLR2004

    expected = pl.DataFrame(
        {
            "component_id": ["0001", "0002", "0003", "0004"],
            "M_zcat": [1.0924, 1.0833, 1.0878, 1.0878],
            "V_des": [68.274, 67.702, 67.988, 67.988],
            "q": [2.797, 2.750, 2.7734, 2.7734],
            "K_ar": [1.0, 1.0, 0.9517, 0.9517],
        }
    )

    for component_id in expected["component_id"]:
        assert isclose(
            structure.results.filter(pl.col("component_id") == component_id)["M_zcat"][
                0
            ],
            expected.filter(pl.col("component_id") == component_id)["M_zcat"][0],
            rel_tol=1e-4,
        )
        assert isclose(
            structure.results.filter(pl.col("component_id") == component_id)["V_des"][
                0
            ],
            expected.filter(pl.col("component_id") == component_id)["V_des"][0],
            rel_tol=1e-4,
        )
        assert isclose(
            structure.results.filter(pl.col("component_id") == component_id)["q"][0],
            expected.filter(pl.col("component_id") == component_id)["q"][0],
            rel_tol=1e-4,
        )
        assert isclose(
            structure.results.filter(pl.col("component_id") == component_id)["K_ar"][0],
            expected.filter(pl.col("component_id") == component_id)["K_ar"][0],
            rel_tol=1e-4,
        )
