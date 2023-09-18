"""
Contains tests for the geotech module.
"""

from math import isclose, radians

import numpy as np
import pytest

from utilityscripts.geotech import (
    boussinesq_patch_sigma_x,
    boussinesq_patch_sigma_z,
    boussinesq_point_sigma_z,
    brinch_hansen_kqz,
    q_ult,
)


def test_q_ult():
    """
    Basic test of the q_ult method against example 8.1 in Craig's Soil Mechanics
    """

    B = 2.25
    D = 1.5

    c = 0
    gamma = 18

    N_c = 1
    N_gamma = 67
    N_q = 49

    s_gamma = 0.8

    q = gamma * D

    expected = 2408
    actual = q_ult(
        c=c, N_c=N_c, gamma=gamma, q=q, N_q=N_q, B=B, N_gamma=N_gamma, s_gamma=s_gamma
    )

    assert isclose(expected, actual, rel_tol=0.001)


@pytest.mark.parametrize(
    "Q, z, r, expected, tol",
    [
        (1000, 5, 0, 19.10, 0.0001),
        (225, 0.001, 0, 1e8, 0.1),  # see J Bowles
        (225, 0.6, 0, 298, 0.005),  # see J Bowles
        (225, 1.2, 0, 74.5, 0.005),  # see J Bowles
        (225, 3.0, 0, 11.9, 0.005),  # see J Bowles
        (500, 2, 2, 10.5, 0.005),  # see J Bowles
        (1000, 2, 1, 68.3, 0.005),  # see J Bowles
        (1500, 5, 0, 29, 0.05),  # see Craig's soil mechanics
    ],
)
def test_boussinesq_point_sigma_z(Q, z, r, expected, tol):
    """
    Test Boussinesq's method based on values calculated at:

    https://testbook.com/civil-engineering/boussinesqs-equation-definition-and-hypothesis

    and in J. Bowles 5ed.

    """

    actual = boussinesq_point_sigma_z(Q=Q, z=z, r=r)

    assert isclose(expected, actual, rel_tol=tol)


@pytest.mark.parametrize(
    "Q, z, r, x_patch, y_patch, expected, tol",
    [
        (1000, 5, 0, 0.00001, 0.00001, 19.10, 0.0001),
        (225, 0.001, 0, 0.00001, 0.00001, 1e8, 0.1),
        (225, 0.6, 0, 0.00001, 0.00001, 298, 0.005),
        (225, 1.2, 0, 0.00001, 0.00001, 74.5, 0.005),
        (225, 3.0, 0, 0.00001, 0.00001, 11.9, 0.005),
        (500, 2, 2, 0.00001, 0.00001, 10.5, 0.005),
        (1000, 2, 1, 0.00001, 0.00001, 68.3, 0.005),
        (1500, 5, 0, 2, 2, 27, 0.01),
        (
            300 * 6 * 3,
            3,
            3,
            3,
            6,
            44,
            0.005,
        ),  # additional test from Craig's soil mechanics.
        (
            250 * 2 * 100,
            3,
            0,
            2,
            100,
            99,
            0.005,
        ),  # additional test from Craig's soil mechanics
    ],
)
@pytest.mark.filterwarnings("ignore:warning")
def test_boussinesq_patch_sigma_z(Q, z, r, x_patch, y_patch, expected, tol):
    """
    Test the Boussinesq patch method against the point loads tested above,
    using a very small patch area.
    """

    x_patch = x_patch
    y_patch = y_patch

    q = Q / (x_patch * y_patch)

    actual = boussinesq_patch_sigma_z(
        q=q, x=r, y=0, z=z, x_patch=x_patch, y_patch=y_patch, n_int_x=100, n_int_y=100
    )

    assert isclose(expected, actual, rel_tol=tol)


@pytest.mark.parametrize(
    "v, expected, tol", [(0.3, 36.967, 0.04), (0.5, 52.672, 0.03), (1.0, 91.935, 0.065)]
)
@pytest.mark.filterwarnings("ignore:warning")
def test_boussinesq_patch_sigma_x(v, expected, tol):
    """
    Test the boussinesq patch method for horizontal stress against example from
    J Bowles (fig E11.8a)

    Note that Bowles' results may have an error in them for the case of v = 1.0,
    where the result calculated by this method is higher by about 6% than
    Bowles calculates.
    """

    height = 7.5
    n_points = 10

    z = np.linspace(0, height, n_points + 1)
    z = (z[1:] + z[:-1]) / 2

    delta_z = height / n_points

    sigma_x = boussinesq_patch_sigma_x(
        q=200,
        x=4,
        y=0,
        z=z,
        v=v,
        x_patch=2,
        y_patch=4,
    )

    actual = np.sum(sigma_x * delta_z)

    assert isclose(expected, actual, rel_tol=tol)


@pytest.mark.parametrize(
    "b, z, phi, expected",
    [
        (1, 1, 5, 0.500),
        (1, 20, 5, 0.725),
        (1, 20000, 5, 0.82),
        (1, 1, 30, 6.0),
        (1, 20, 30, 14),
        (1, 20000, 30, 17.7),
        (1, 1, 45, 22),
        (1, 20, 45, 75),
        (1, 20000, 45, 222),
    ],
)
def test_bh_kqz(b, z, phi, expected):
    """
    Test the Brinch-Hansen Kqz method against values graphed in the
    Australian Structural Engineer's Guidebook.

    Values are fairly imprecise because they are read off a graph.
    """

    phi = radians(phi)

    actual = brinch_hansen_kqz(z=z, b=b, phi=phi)

    assert isclose(expected, actual, rel_tol=0.1)
