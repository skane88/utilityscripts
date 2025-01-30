"""
A file to store results along with some reporting information.
"""

from typing import Any


class Result:
    """
    Class to store results along with the equation, inputs,
    and metadata used to generate it. The Result object will also produce formatted
    strings for use in reports etc.

    Notes
    -----
    It is intended that native operations on the Result object should act on
    the stored .result attribute as though it were a stand-alone value.
    e.g. Result * 2 should be equivalent to Result.result * 2 and so-on.
    Initially the basic arithmetic operations will be implemented and over time
    additional operations may be added. Be-aware that if the .result attribute
    does not support an operation then an error may be raised.

    Usage
    -----
    >>> r = Result(1, eqn={"x": 1}, inputs={"x": 1}, metadata={"source": "test"})
    >>> r * 2
    2
    """

    def __init__(
        self,
        result: Any,
        eqn: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._result = result
        self._eqn = eqn
        self._inputs = inputs
        self._metadata = metadata

    @property
    def result(self) -> Any:
        """
        The result stored in the Result object.
        """

        return self._result

    @property
    def eqn(self) -> dict[str, Any] | None:
        """
        The equation used to generate the results
        """

        return self._eqn

    @property
    def inputs(self) -> dict[str, Any] | None:
        """
        The inputs to the equation.
        """

        return self._inputs

    @property
    def metadata(self) -> dict[str, Any] | None:
        """
        The metadata stored in the Result object.
        """

        return self._metadata

    def __str__(self):
        if isinstance(self._result, str):
            return self._result
        return str(self._result)

    def __bool__(self):
        if isinstance(self._result, bool):
            return self._result
        return bool(self._result)

    def __int__(self):
        if isinstance(self._result, int):
            return self._result
        return int(self._result)

    def __float__(self):
        if isinstance(self._result, float):
            return self._result
        return float(self._result)

    def __bytes__(self):
        if isinstance(self._result, bytes):
            return self._result
        return bytes(self._result)

    def __complex__(self):
        if isinstance(self._result, complex):
            return self._result
        return complex(self._result)

    def __repr__(self):
        return (
            f"Result(result={self._result}, "
            + f"eqn={self._eqn}, "
            + f"inputs={None if self._inputs is None else f'{len(self._inputs)} inputs'}, "
            + f"metadata={None if self._metadata is None else f'{len(self._metadata)} items'})"
        )
