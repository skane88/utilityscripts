"""
Utility script to convert old imperial sections to modern metric equivalents.
"""

import re
from fractions import Fraction

from humre import (
    DIGIT,
    DOUBLE_QUOTE,
    PERIOD,
    QUOTE,
    WHITESPACE,
    chars,
    either,
    exactly,
    group,
    negative_lookahead,
    one_or_more,
    one_or_more_group,
    optional_group,
    zero_or_more,
    zero_or_more_group,
)

FEET_TO_INCHES = 12
INCHES_TO_M = 0.0254

FEET = group(group(one_or_more(DIGIT)) + group(QUOTE))

WHOLE_INCHES = group(one_or_more(DIGIT)) + negative_lookahead(PERIOD)
DECIMAL_INCHES = group(one_or_more(DIGIT) + PERIOD + one_or_more(DIGIT))
FRACTIONAL_INCHES = group(
    group(one_or_more(DIGIT))
    + zero_or_more(WHITESPACE)
    + group(f"{one_or_more(DIGIT)}/{one_or_more(DIGIT)}")
)

INCHES = group(either(WHOLE_INCHES, DECIMAL_INCHES, FRACTIONAL_INCHES)) + group(
    DOUBLE_QUOTE
)

FEET_AND_INCHES = (
    optional_group(FEET)
    + optional_group(zero_or_more(WHITESPACE))
    + optional_group(INCHES)
)


def feet_inches_to_m(imperial: str) -> float:
    """
    Convert a text string of feet & inches (e.g. 6' or 32" or 1' 1 1/4") into m.

    :param imperial: the text to convert.
    """

    if r"\'" in imperial or r"\"" in imperial:
        raise ValueError(
            "Escaped \\\" or \\' found in input string. "
            + "Did you mean to use a raw string, or to not escape the character?"
        )

    feet_and_inches = re.compile(FEET_AND_INCHES).match(imperial)

    is_feet = feet_and_inches[1]

    feet = float(feet_and_inches[3]) if is_feet else 0

    if feet_and_inches[6]:

        if feet_and_inches[10]:
            inches = float(sum(Fraction(s) for s in feet_and_inches[10].split()))
        elif feet_and_inches[9]:
            inches = float(feet_and_inches[9])
        else:
            inches = float(feet_and_inches[7])
    else:
        inches = 0

    return (feet * FEET_TO_INCHES + inches) * INCHES_TO_M
