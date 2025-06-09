"""
A file to store results along with some reporting information.
"""

from typing import Any

from jinja2 import Environment

latex_env = Environment(
    block_start_string="<<b",
    block_end_string=">>",
    variable_start_string="<<",
    variable_end_string=">>",
    comment_start_string="<<#",
    comment_end_string=">>",
    autoescape=True,
)

text_env = Environment(autoescape=True)

default_text_template = """{{ variable }}
========================================
{%if description is defined%}{{ description }}{% endif %}
{% for variable in variables.values() %}{{ variable.variable }} = {{ variable.str_value }}{% endfor %}
{%if eqn is not none %}{{ eqn }} {% else %}value{% endif %}={{ value }}
"""


class Variable:
    def __init__(
        self,
        variable: str,
        value: Any,
        units: str | None = None,
        no_places: int | None = None,
    ):
        self._variable = variable
        self._value = value
        self._units = units
        self._no_places = no_places

    @property
    def variable(self) -> str:
        return self._variable

    @property
    def value(self) -> Any:
        return self._value

    @property
    def units(self) -> str | None:
        return self._units

    @property
    def no_places(self) -> int | None:
        return self._no_places

    @property
    def str_value(self) -> str:
        return (
            f"{self.value:.{self.no_places}f}"
            if self.no_places is not None
            else f"{self.value}"
        ) + (f" {self.units}" if self.units is not None else "")

    def __repr__(self):
        return f"Variable: {self.variable}={self.value}"


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
    >>> r.val
    1
    """

    def __init__(
        self,
        value: Any,
        *,
        description: str | None = None,
        variable: str | None = None,
        eqn: str | None = None,
        data: dict[str, Any] | None = None,
        units: str | None = None,
        no_places: int | None = None,
        text_template: str | None = None,
        latex_template: str | None = None,
    ):
        """
        Initialise a Result object.

        Parameters
        ----------
        value : Any
            The result stored in the Result object.
        description : str | None, optional
            A description of the Result object.
        variable : str | None, optional
            The variable name of the Result object.
        eqn : str | None, optional
            The equation used to generate the result.
        data : dict[str, Any] | None, optional
            Any extra data stored in the Result object.
        text_template : str | None, optional
            A template for the text representation of the Result object.
        latex_template : str | None, optional
            A template for the LaTeX representation of the Result object.
        """

        self._value = value
        self._description = description
        self._variable = variable
        self._eqn = eqn
        self._data = data
        self._units = units
        self._no_places = no_places
        self._text_template = text_template
        self._latex_template = latex_template

    @property
    def value(self) -> Any:
        """
        The result stored in the Result object.
        """

        return self._value

    @property
    def eqn(self) -> str | None:
        """
        The equation used to generate the result.
        """

        return self._eqn

    @property
    def description(self) -> str:
        """
        The description of the Result object.
        """

        if self._description is None:
            return ""

        return self._description

    @property
    def variable(self) -> str | None:
        """
        The variable name of the Result object.
        """

        return self._variable

    @property
    def data(self) -> dict[str, Any] | None:
        """
        Any extra data stored in the Result object.
        """

        return self._data

    @property
    def units(self) -> str | None:
        """
        The units of the result.
        """

        return self._units

    @property
    def no_places(self) -> int | None:
        """
        The number of decimal places to display for the result.
        """

        return self._no_places

    @property
    def variables(self) -> dict[str, Variable]:
        """
        Variable objects based on the data stored in the object.
        """

        variables = {}

        if self._data is None:
            return {}

        for key, value in self._data.items():
            if isinstance(value, Variable):
                variables[key] = value

            else:
                variables[key] = Variable(key, value)

        return variables

    @property
    def plain_string(self) -> str:
        template = (
            default_text_template
            if self._text_template is None
            else self._text_template
        )

        return text_env.from_string(template).render(
            description=self.description,
            variable=self.variable,
            eqn=self.eqn,
            variables=self.variables,
            value=self.value,
        )

    def __repr__(self):
        return f"Result: {self.variable}={self.value}"


class DeprecatedResult(Result):
    def __str__(self):
        if isinstance(self._value, str):
            return self._value
        return str(self._value)

    def __bool__(self):
        if isinstance(self._value, bool):
            return self._value
        return bool(self._value)

    def __int__(self):
        if isinstance(self._value, int):
            return self._value
        return int(self._value)

    def __float__(self):
        if isinstance(self._value, float):
            return self._value
        return float(self._value)

    def __bytes__(self):
        if isinstance(self._value, bytes):
            return self._value
        return bytes(self._value)

    def __complex__(self):
        if isinstance(self._value, complex):
            return self._value
        return complex(self._value)

    def __neg__(self):
        if hasattr(self._value, "__neg__"):
            return self._value.__neg__()
        return NotImplemented

    def __pos__(self):
        if hasattr(self._value, "__pos__"):
            return self._value.__pos__()
        return NotImplemented

    def __invert__(self):
        if hasattr(self._value, "__invert__"):
            return self._value.__invert__()
        return NotImplemented

    def __format__(self, format_spec: str) -> str:
        if hasattr(self._value, "__format__"):
            return self._value.__format__(format_spec)
        return NotImplemented

    def __add__(self, other):
        if hasattr(self._value, "__add__"):
            return self._value + other
        if hasattr(other, "__radd__"):
            return self._value + other
        return NotImplemented

    def __radd__(self, other):
        if hasattr(self._value, "__radd__"):
            return other + self._value
        if hasattr(other, "__add__"):
            return other + self._value

        return NotImplemented

    def __sub__(self, other):
        if hasattr(self._value, "__sub__"):
            return self._value - other
        if hasattr(other, "__rsub__"):
            return self._value - other

        return NotImplemented

    def __rsub__(self, other):
        if hasattr(self._value, "__rsub__"):
            return other - self._value
        if hasattr(other, "__sub__"):
            return other - self._value

        return NotImplemented

    def __mul__(self, other):
        if hasattr(self._value, "__mul__"):
            return self._value * other

        if hasattr(other, "__rmul__"):
            return self._value * other

        return NotImplemented

    def __rmul__(self, other):
        if hasattr(self._value, "__rmul__"):
            return other * self._value
        if hasattr(other, "__mul__"):
            return other * self._value

        return NotImplemented

    def __truediv__(self, other):
        if hasattr(self._value, "__truediv__"):
            return self._value / other
        if hasattr(other, "__rtruediv__"):
            return self._value / other

        return NotImplemented

    def __rtruediv__(self, other):
        if hasattr(self._value, "__rtruediv__"):
            return other / self._value
        if hasattr(other, "__truediv__"):
            return other / self._value
        return NotImplemented

    def __floordiv__(self, other):
        if hasattr(self._value, "__floordiv__"):
            return self._value // other
        if hasattr(other, "__rfloordiv__"):
            return self._value // other
        return NotImplemented

    def __rfloordiv__(self, other):
        if hasattr(self._value, "__rfloordiv__"):
            return other // self._value
        if hasattr(other, "__floordiv__"):
            return other // self._value
        return NotImplemented

    def __pow__(self, other):
        if hasattr(self._value, "__pow__"):
            return self._value**other
        if hasattr(other, "__rpow__"):
            return self._value**other
        return NotImplemented

    def __rpow__(self, other):
        if hasattr(self._value, "__rpow__"):
            return other**self._value
        if hasattr(other, "__pow__"):
            return other**self._value
        return NotImplemented

    def report(
        self,
        result_format: str | None = None,
        input_formats: dict[str, str] | None = None,
    ) -> str:
        """
        A short report on the result.
        """

        result_str = self.description + "\n" if self.description is not None else ""

        if self.inputs is not None:
            for i in self.inputs:
                if input_formats is None or i not in input_formats:
                    input_val = f"{self.inputs[i]}"
                else:
                    input_val = f"{self.inputs[i]:{input_formats[i]}}"

                result_str += f"{i}: " + input_val + "\n"

        result_str += "Equation:" + self.eqn + "\n" if self.eqn is not None else ""

        result_value = (
            str(self.value)
            if result_format is None
            else f"{self.value:{result_format}}"
        )

        result_str += self.variable if self.variable is not None else ""
        result_str += "=" + result_value

        return result_str
