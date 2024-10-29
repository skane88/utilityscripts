"""
Module to contain tests for code in steel.py
"""

from utilityscripts.steel.steel import local_thickness_reqd


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
