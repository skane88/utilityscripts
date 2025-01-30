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
