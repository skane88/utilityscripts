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
    either,
    group,
    negative_lookahead,
    one_or_more,
    optional_group,
    zero_or_more,
)

FEET_TO_INCHES = 12
INCHES_TO_M = 0.0254

FEET = group(group(one_or_more(DIGIT)) + group(QUOTE))

WHOLE_INCHES = group(one_or_more(DIGIT)) + negative_lookahead(PERIOD)
DECIMAL_INCHES = group(one_or_more(DIGIT) + PERIOD + one_or_more(DIGIT))
FRACTIONAL_INCHES = group(
    optional_group(one_or_more(DIGIT))
    + optional_group(zero_or_more(WHITESPACE))
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


def ft_in_to_m(feet: float = 0, inches: float = 0) -> float:
    """
    Converts ft and inches into metres.

    :param feet: The feet to convert.
    :param inches: The inches to convert.
    """

    return (feet * FEET_TO_INCHES + inches) * INCHES_TO_M


def ft_in_str_to_m(imperial: str) -> float:
    """
    Convert a text string of feet & inches (e.g. 6' or 32" or 1' 1 1/4") into m.

    :param imperial: the text to convert.
    """

    if r"\'" in imperial or r"\"" in imperial:
        raise ValueError(
            "Escaped \\\" or \\' found in input string. "
            + "Did you mean to use a raw string, or to not escape the character?"
        )

    if imperial.count("'") > 1 or imperial.count('"') > 1:
        raise ValueError("Multiple occurrences of the foot (') or inch symbols (\").")

    imperial = imperial.replace(" / ", "/")

    feet_and_inches = re.compile(FEET_AND_INCHES).match(imperial)

    base = feet_and_inches[0]
    foot_value = feet_and_inches[3]
    whole_inches = feet_and_inches[7]
    decimal_inches = feet_and_inches[9]
    fractional_inches = feet_and_inches[10]

    foot_part = feet_and_inches[1]

    inch_part = feet_and_inches[6]

    if base == "":
        raise ValueError("Could not parse the input string into feet & inches.")

    feet = float(foot_value) if foot_part else 0
    if inch_part:
        if fractional_inches:
            inches = float(sum(Fraction(s) for s in fractional_inches.split()))
        elif decimal_inches:
            inches = float(decimal_inches)
        else:
            inches = float(whole_inches)
    else:
        inches = 0

    return ft_in_to_m(feet=feet, inches=inches)
