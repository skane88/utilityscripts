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
        description: str | None = None,
        variable: str | None = None,
        eqn: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._result = result
        self._description = description
        self._variable = variable
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
    def description(self) -> str | None:
        """
        The description of the Result object.
        """

        return self._description

    @property
    def variable(self) -> str | None:
        """
        The variable name of the Result object.
        """

        return self._variable

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

    def __neg__(self):
        if hasattr(self._result, "__neg__"):
            return self._result.__neg__()
        raise NotImplementedError(
            f"{self.result!r} does not support the __neg__ operation."
        )

    def __pos__(self):
        if hasattr(self._result, "__pos__"):
            return self._result.__pos__()
        raise NotImplementedError(
            f"{self.result!r} does not support the __pos__ operation."
        )

    def __invert__(self):
        if hasattr(self._result, "__invert__"):
            return self._result.__invert__()
        raise NotImplementedError(
            f"{self.result!r} does not support the __invert__ operation."
        )

    def __format__(self, format_spec: str) -> str:
        if hasattr(self._result, "__format__"):
            return self._result.__format__(format_spec)
        raise NotImplementedError(
            f"{self.result!r} does not support the __format__ operation."
        )

    def __mul__(self, other):
        if hasattr(self._result, "__mul__"):
            return self._result.__mul__(other)
        raise NotImplementedError(
            f"{self.result!r} does not support the __mul__ operation."
        )

    def __rmul__(self, other):
        if hasattr(self._result, "__rmul__"):
            return self._result.__rmul__(other)
        raise NotImplementedError(
            f"{self.result!r} does not support the __rmul__ operation."
        )

    def report(
        self,
        result_format: str | None = None,
        input_formats: dict[str, str] | None = None,
    ) -> str:
        """
        A short report on the result.
        """

        result_str = self.description + "\n" if self.description is not None else ""

        if self._inputs is not None:
            for i in self._inputs:
                if input_formats is None or i not in input_formats:
                    input_val = f"{self.inputs[i]}"
                else:
                    input_val = f"{self.inputs[i]:{input_formats[i]}}"

                result_str += f"{i}: " + input_val + "\n"

        result_str += "Equation:" + self.eqn + "\n" if self.eqn is not None else ""

        result_value = (
            str(self.result)
            if result_format is None
            else f"{self.result:{result_format}}"
        )

        if self.variable is None:
            result_str += result_value
        else:
            result_str += self.variable + "=" + result_value

        return result_str

    def __repr__(self):
        return (
            f"Result(result={self._result}, "
            + f"eqn={self._eqn}, "
            + "inputs="
            + f"{None if self._inputs is None else f'{len(self._inputs)} inputs'}, "
            + "metadata="
            + f"{None if self._metadata is None else f'{len(self._metadata)} items'})"
        )
