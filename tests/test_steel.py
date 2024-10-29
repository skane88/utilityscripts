"""
Module to contain tests for code in steel.py
"""

from math import isclose

import pytest

from utilityscripts.steel.as4100 import k_t
from utilityscripts.steel.steel import local_thickness_reqd


@pytest.mark.parametrize(
    "restraint_code, d_1, length, t_f, t_w, n_w, expected",
    [
        ("PP", 0.2060, 4.00, 0.0120, 0.0065, 1, 1.081012),
        ("PF", 0.2834, 10.0, 0.0118, 0.0067, 1, 1.019352),
        ("FF", 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
        ("FL", 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
        ("LF", 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
        ("LL", 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
        ("FU", 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
        ("UF", 0.2834, 10.0, 0.0118, 0.0067, 1, 1.0),
    ],
)
def test_k_t(restraint_code, d_1, length, t_f, t_w, n_w, expected):
    """
    Test the k_t method.
    """

    k_t_calc = k_t(
        restraint_code=restraint_code, d_1=d_1, length=length, t_f=t_f, t_w=t_w, n_w=n_w
    )

    assert isclose(k_t_calc, expected, rel_tol=1e-6)


def test_thickness_reqd():
    """
    Test the method that calculates the thickness required for a plate with a point load.
    """

    force = 10e3
    f_y = 250e6
    phi = 0.9

    assert local_thickness_reqd(
        point_force=force, f_y=f_y, phi=phi
    ) == local_thickness_reqd(
        point_force=force, f_y=f_y, phi=phi, width_lever=(0.0, 0.0)
    )

    assert (
        local_thickness_reqd(point_force=force, f_y=f_y, phi=phi)
        == 0.009428090415820633  # noqa: PLR2004
    )

    width_lever = (0.022, 0.035)

    assert (
        local_thickness_reqd(
            point_force=force, f_y=f_y, phi=phi, width_lever=width_lever
        )
        == 0.008223919396586149  # noqa: PLR2004
    )
