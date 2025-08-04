"""
Tests of the AS4600 module.
"""

import pytest

from utilityscripts.steel.as4600 import S5342ConnType, t5_3_4_2_alpha


@pytest.mark.parametrize(
    "conn_type, expected",
    [
        (S5342ConnType.SINGLE_SHEAR_BOTH_WASHERS, 1.00),
        (S5342ConnType.SINGLE_SHEAR_ONE_WASHER, 0.75),
        (S5342ConnType.SINGLE_SHEAR_OVERSIZED_HOLES, 0.70),
        (S5342ConnType.SINGLE_SHEAR_OVERSIZED_HOLES_PERPENDICULAR, 0.55),
        (S5342ConnType.DOUBLE_SHEAR_INNER, 1.33),
        (S5342ConnType.DOUBLE_SHEAR_INNER_OVERSIZED_HOLES, 1.10),
        (S5342ConnType.DOUBLE_SHEAR_INNER_OVERSIZED_HOLES_PERPENDICULAR, 0.90),
    ],
)
def test_t5_3_4_2_alpha(conn_type, expected):
    assert t5_3_4_2_alpha(conn_type) == expected
