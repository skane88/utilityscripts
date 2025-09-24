"""
Test functions in the as4678 module
"""

from math import isclose, radians

import pytest

from utilityscripts.geotech.as4678 import s5_2_2_phi_star


@pytest.mark.parametrize(
    "phi_u_phi, phi, expected",
    [
        (0.85, 30, radians(26.14)),
    ],
)
def test_s5_2_2_phi_star(phi_u_phi, phi, expected):
    assert isclose(
        s5_2_2_phi_star(phi_u_phi=phi_u_phi, phi=phi), expected, rel_tol=1e-3
    )
