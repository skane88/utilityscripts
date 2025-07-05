"""
A file to store results along with some reporting information.
"""

from typing import Any

from jinja2 import Environment


class ResultError(Exception):
    pass


latex_env = Environment(
    block_start_string=r"\BLOCK{",  # update some control characters to suit latex
    block_end_string=r"}",
    variable_start_string=r"\VAR{",
    variable_end_string=r"}",
    comment_start_string=r"\#{",
    comment_end_string=r"}",
    trim_blocks=True,
    autoescape=True,
)

text_env = Environment(autoescape=True)

default_text_template = (
    "{{ variable }}\n"
    + "----------------------------------------\n"
    + "{% if description is defined%}{{ description }}\n{% endif %}"
    + "{% for variable in variables.values() %}{{ variable.symbol }} = {{ variable.report_string }}\n{% endfor %}"
    + "{% if eqn is not none %}{{ eqn }}{% else %}value{% endif %} = {{ str_value }}"
)

default_latex_template = (
    r"\VAR{variable} \newline"
    + r"\underline{\hspace{5cm}} \newline"
    + r"\BLOCK{if description is defined} \VAR{description} \newline \BLOCK{ endif }"
    + r"\BLOCK{for variable in variables.values()} \VAR{variable.symbol} = \VAR{variable.report_string} \newline \BLOCK{ endfor }"
    + r"\BLOCK{if eqn is not none} \VAR{eqn} \BLOCK{ else } value \BLOCK{ endif } = \VAR{ str_value }"
)


class Variable:
    """
    Class to store a simple variable object. Designed to package together a value,
    a symbol and some information about units and how to display it.

    Notes
    -----
    - Intended to be for reporting purposes only. The units functionality is not and
    will not replace a units package like pint or unyt.
    """

    def __init__(
        self,
        value: Any,
        *,
        symbol: str | None = None,
        units: str | None = None,
        fmt_string: str | None = None,
    ):
        self._value = value
        self._symbol = symbol
        self._units = units
        self._fmt_string = fmt_string

        if (
            self._fmt_string is not None
            and "%" in self._fmt_string
            and self._units is not None
        ):
            raise ResultError("Using units with % formatting does not make sense.")

    @property
    def value(self) -> Any:
        return self._value

    @property
    def symbol(self) -> str | None:
        return self._symbol

    @property
    def units(self) -> str:
        return self._units

    @property
    def fmt_string(self) -> str:
        return self._fmt_string

    @property
    def latex_string(self) -> str:
        # TODO: update this to use python's format strings to build a latex
        #  string - in particular, need a function to use the .#e format string
        #  to determine the scientific notation.

        unit_str = f" \\text{{{self.units}}}" if self.units else ""
        symbol_str = f"\\text{{{self.symbol}}} = " if self.symbol else ""

        value_str = (
            f"{self.value:{self.fmt_string}}"
            if self.fmt_string is not None
            else f"{self.value}"
        )

        if "e" in value_str.lower():
            mantissa, exponent = value_str.lower().split("e")
            exponent = int(exponent)

            value_str = f"{mantissa} \\times 10^{{{exponent}}}"

        value_str = value_str.replace("%", "\\%")

        return symbol_str + value_str + unit_str

    def _repr_mimebundle_(self, include=None, exclude=None):
        """
        Provide a mimebundle as required by
        https://ipython.readthedocs.io/en/stable/config/integrating.html
        to allow for simple outputs from Jupyter notebooks and similar.

        Notes
        -----
        - the '$' signs are only added to the latex output in this method
        so that the user can get a plain latex output from the other methods.
        """

        return {
            "text/plain": self.__str__(),
            "text/latex": "$" + self.latex_string + "$",
        }

    def __str__(self):
        symbol_str = f"{self.symbol}=" if self.symbol else ""
        value_str = (
            f"{self.value:{self.fmt_string}}"
            if self.fmt_string is not None
            else f"{self.value}"
        )
        unit_str = f" {self.units}" if self.units else ""

        return symbol_str + value_str + unit_str

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.value!r}"
            + f", symbol={self.symbol!r}"
            + f", units={self.units!r}"
            + f", fmt_string={self.fmt_string!r})"
        )


class Result:
    """
    Class to store results along with the equation, inputs,
    and metadata used to generate it. The Result object will also produce formatted
    strings for use in reports etc.
    """

    def __init__(
        self,
        value: Any,
        *,
        description: str | None = None,
        symbol: str | None = None,
        eqn: str | None = None,
        data: dict[str, Any] | None = None,
        units: str = "",
        fmt_string: str = ".3e",
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
        symbol : str | None, optional
            The symbol used for the result.
        eqn : str | None, optional
            The equation used to generate the result.
        data : dict[str, Any] | None, optional
            Any extra data stored in the Result object.
        units : str, optional
            The units of the result.
        fmt_string : str, optional
            The format string used to display the result.
        text_template : str | None, optional
            A template for the text representation of the Result object.
        latex_template : str | None, optional
            A template for the LaTeX representation of the Result object.
        """

        self._value = value
        self._description = description
        self._symbol = symbol
        self._eqn = eqn
        self._data = data
        self._units = units
        self._fmt_string = fmt_string
        self._text_template = text_template
        self._latex_template = latex_template

    @property
    def value(self) -> Any:
        """
        The result stored in the Result object.
        """

        return self._value

    @property
    def str_value(self) -> str:
        """
        The string representation of the result.
        """

        return f"{self.value:{self.fmt_string}}" + f"{self.units}"

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
    def symbol(self) -> str | None:
        """
        The symbol used for the Result object.
        """

        return self._symbol

    @property
    def data(self) -> dict[str, Any] | None:
        """
        Any extra data stored in the Result object.
        """

        return self._data

    @property
    def units(self) -> str:
        """
        The units of the result.
        """

        return self._units

    @property
    def fmt_string(self) -> str:
        """
        The format string used to display the result.
        """

        return self._fmt_string

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
            variable=self.symbol,
            eqn=self.eqn,
            variables=self.variables,
            str_value=self.str_value,
        )

    @property
    def latex_string(self) -> str:
        template = (
            default_latex_template
            if self._latex_template is None
            else self._latex_template
        )

        return latex_env.from_string(template).render(
            description=self.description,
            variable=self.symbol,
            eqn=self.eqn,
            variables=self.variables,
            str_value=self.str_value,
        )

    def __repr__(self):
        return f"Result: {self.symbol}={self.str_value}"


class DeprecatedResult(Result):
    """
    Deprecated result class. One day it may be rolled back into the master Result class,
    possibly through a factory function that builds IntResults, FloatResults etc.

    Notes
    -----
    It is intended that native operations on the Result object should act on
    the stored .result attribute as though it were a stand-alone value.
    e.g. Result * 2 should be equivalent to Result.result * 2 and so-on.
    Initially the basic arithmetic operations will be implemented and over time
    additional operations may be added. Be-aware that if the .result attribute
    does not support an operation then an error may be raised.
    """

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

        result_str += self.symbol if self.symbol is not None else ""
        result_str += "=" + result_value

        return result_str
