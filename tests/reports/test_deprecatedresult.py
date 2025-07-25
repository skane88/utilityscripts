import pytest

from utilityscripts.reports.report import DeprecatedResult


@pytest.mark.parametrize("operation", [str, float, int, bool, complex, bytes])
def test_result_to_str(operation):
    r = DeprecatedResult(1)
    assert operation(r) == operation(1)


def test_result_neg():
    r = DeprecatedResult(1)
    assert -r == -1


def test_result_pos():
    r = DeprecatedResult(1)
    assert +r == +1


def test_result_invert():
    r = DeprecatedResult(1)
    assert ~r == ~1


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 1 + 2),
        (2, 1, 2 + 1),
        ("a", "b", "a" + "b"),
        ("b", "a", "b" + "a"),
        (True, True, 2),
        (True, False, 1),
        (False, False, 0),
    ],
)
def test_result_add(result, other, expected):
    result = DeprecatedResult(result)
    assert result + other == expected


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 1 - 2),
        (2, 1, 2 - 1),
        (1.0, 2.0, 1.0 - 2.0),
        (2.0, 1.0, 2.0 - 1.0),
        (True, True, 0),
        (True, False, 1),
        (False, False, 0),
    ],
)
def test_result_sub(result, other, expected):
    result = DeprecatedResult(result)
    assert result - other == expected


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
    result = DeprecatedResult(result)
    assert result * other == expected
    assert other * result == expected


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 1 / 2),
        (1, 0.5, 1 / 0.5),
    ],
)
def test_result_truediv(result, other, expected):
    result = DeprecatedResult(result)
    assert result / other == expected


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 2 / 1),
        (1, 0.5, 0.5 / 1),
    ],
)
def test_result_rtruediv(result, other, expected):
    result = DeprecatedResult(result)
    assert other / result == expected


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 1 // 2),
        (1, 0.5, 1 // 0.5),
    ],
)
def test_result_floordiv(result, other, expected):
    result = DeprecatedResult(result)
    assert result // other == expected


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 2 // 1),
        (1, 0.5, 0.5 // 1),
    ],
)
def test_result_rfloordiv(result, other, expected):
    result = DeprecatedResult(result)
    assert other // result == expected


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 1**2),
        (1, 0.5, 1**0.5),
    ],
)
def test_result_pow(result, other, expected):
    result = DeprecatedResult(result)
    assert result**other == expected


@pytest.mark.parametrize(
    "result, other, expected",
    [
        (1, 2, 2**1),
        (1, 0.5, 0.5**1),
    ],
)
def test_result_rpow(result, other, expected):
    result = DeprecatedResult(result)
    assert other**result == expected
