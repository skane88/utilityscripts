"""
Some tests for the concrete module.
"""

from math import isclose

import pytest

from utilityscripts.concrete import mesh_mass, reo_prop

REL_TOL = 0.01


@pytest.mark.parametrize(
    "bar_spec,area",
    [
        ("N20", 314),
        ("2-N20", 628),
        ("1-N36", 1020),
        ("9-N32", 7200),
        ("L12", 113),
        ("R12", 113),
        ("N20-200", 1570),
        ("N16-380", 526),
        ("N36-160", 6375),
    ],
)
def test_bar_area(bar_spec, area):
    assert isclose(reo_prop(bar_spec).main_area_total, area, rel_tol=REL_TOL)


@pytest.mark.parametrize(
    "bar_spec,area,main_direction",
    [
        ("SL82", 227, True),
        ("SL82", 227, False),
        ("RL1218", 1112, True),
        ("RL1218", 227, False),
        ("F1218", 1227, True),
        ("F1218", 251, False),
    ],
)
def test_bar_area_mesh(bar_spec, area, main_direction):
    assert isclose(
        (
            reo_prop(bar_spec).main_area_total
            if main_direction
            else reo_prop(bar_spec).secondary_area_total
        ),
        area,
        rel_tol=REL_TOL,
    )


@pytest.mark.parametrize(
    "bar_spec,width,main_direction,expected",
    [
        (
            "SL82",
            1200,
            True,
            {
                "single_bar_area": 45.4,
                "total_area": 272,
                "area_unit_width": 227,
                "no_bars": 6.0,
                "width": 1200,
            },
        ),
        (
            "SL82",
            1200,
            False,
            {
                "single_bar_area": 45.4,
                "total_area": 272,
                "area_unit_width": 227,
                "no_bars": 6.0,
                "width": 1200,
            },
        ),
    ],
)
def test_bar_area_full(bar_spec, width, main_direction, expected):
    return_vals = reo_prop(
        bar_spec=bar_spec,
        main_width=width if main_direction else 1000.0,
        secondary_width=1000.0 if main_direction else width,
    )

    assert isclose(
        return_vals.main_bar_area if main_direction else return_vals.secondary_bar_area,
        expected["single_bar_area"],
        rel_tol=REL_TOL,
    )
    assert isclose(
        (
            return_vals.main_area_total
            if main_direction
            else return_vals.secondary_area_total
        ),
        expected["total_area"],
        rel_tol=REL_TOL,
    )
    assert isclose(
        (
            return_vals.main_area_unit
            if main_direction
            else return_vals.secondary_area_unit
        ),
        expected["area_unit_width"],
        rel_tol=REL_TOL,
    )
    assert isclose(
        return_vals.no_main if main_direction else return_vals.no_secondary,
        expected["no_bars"],
        rel_tol=REL_TOL,
    )
    assert isclose(
        return_vals.main_width if main_direction else return_vals.secondary_width,
        expected["width"],
        rel_tol=REL_TOL,
    )


@pytest.mark.parametrize(
    "main_diameter,secondary_diameter,main_spacing,secondary_spacing,expected",
    [
        (0.0095, 0.0095, 0.200, 0.200, 5.6),  # SL102
        (0.0095, None, 0.200, None, 5.6),  # SL102, test default parameters
        (0.0095, 0.0095, 0.200, None, 5.6),
        (0.0095, None, 0.200, 0.200, 5.6),
        (0.0119, 0.0076, 0.100, 0.200, 10.5),  # RL1218
        (0.0076, 0.0076, 0.100, 0.200, 5.3),  # RL818
    ],
)
def test_mesh_mass(
    main_diameter, secondary_diameter, main_spacing, secondary_spacing, expected
):
    mass = mesh_mass(
        main_diameter=main_diameter,
        secondary_diameter=secondary_diameter,
        main_spacing=main_spacing,
        secondary_spacing=secondary_spacing,
    )

    assert isclose(mass, expected, rel_tol=REL_TOL)
