"""
Utility script to convert old imperial sections to modern metric equivalents.
"""

import re

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

INCHES = either(WHOLE_INCHES, DECIMAL_INCHES, FRACTIONAL_INCHES) + DOUBLE_QUOTE

FEET_AND_INCHES = (
    optional_group(FEET) + group(zero_or_more(WHITESPACE)) + optional_group(INCHES)
)


def feet_inches_to_m(section):

    feet = re.compile(FEET)
    inches = re.compile(INCHES)
    feet_and_inches = re.compile(FEET_AND_INCHES)

    feet = feet.match(section)
    inches = inches.match(section)
    feet_and_inches = feet_and_inches.match(section)

    is_feet = feet_and_inches[1]

    feet = float(feet_and_inches[3]) if is_feet else 0

    is_inches = feet_and_inches[6]

    f = feet_and_inches[5]
    g = feet_and_inches[6]
    h = feet_and_inches[7]
    i = feet_and_inches[8]
    j = feet_and_inches[9]
    k = feet_and_inches[10]
    l = feet_and_inches[11]

    if is_inches:

        if feet_and_inches[7]:
            inches = float(feet_and_inches[7])
        if feet_and_inches[8]:
            inches = float(feet_and_inches[8])

    else:
        inches = 0

    return (feet * FEET_TO_INCHES + inches) * INCHES_TO_M
