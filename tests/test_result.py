"""
Test the Result class.
"""

import pytest

from utilityscripts.result import Result


@pytest.mark.parametrize("operation", [str, float, int, bool, complex, bytes])
def test_result_to_str(operation):
    r = Result(1, eqn={"x": 1}, inputs={"x": 1}, metadata={"source": "test"})
    assert operation(r) == operation(1)


def test_result_neg():
    r = Result(1, eqn={"x": 1}, inputs={"x": 1}, metadata={"source": "test"})
    assert -r == -1


def test_result_pos():
    r = Result(1, eqn={"x": 1}, inputs={"x": 1}, metadata={"source": "test"})
    assert +r == +1


def test_result_invert():
    r = Result(1, eqn={"x": 1}, inputs={"x": 1}, metadata={"source": "test"})
    assert ~r == ~1


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 1 * 2),
        (1, 0.5, 1 * 0.5),
        (1, -1, 1 * -1),
        (1, -0.5, 1 * -0.5),
        (1, 0, 1 * 0),
        ("a", 2, "a" * 2),
        (2, "a", 2 * "a"),
    ],
)
def test_result_mul(result, other, expected):
    result = Result(result)
    assert result * other == expected
