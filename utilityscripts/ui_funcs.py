"""
File to contain basic UI functions.
"""

import contextlib
from pathlib import Path
from tkinter import Tk, filedialog

yes_set = {"y", "Y", "Yes", "YES", "yes"}
no_set = {"n", "N", "No", "NO", "no"}
quit_set = {"quit", "q", "Q", "Quit"}


def get_folder(*, save: bool = True, type_prompt: str) -> Path | None:
    """
    Gets a folder location.
    :param save: if True, the strings etc. will be for a save operation, if False for a
        load operation.
    :param type_prompt: A string with the type of file to save / get from the folder.
        Used for the user prompt string.
        e.g. "photos" or "excel files" or "reports"
    :return: The chosen folder. If nothing is chosen, return None.
    """

    if save:
        print(f"Choose a folder to save the {type_prompt} to:")
    else:
        print(f"Choose a folder to load the {type_prompt} from:")

    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory()
    root.destroy()

    return None if folder == "" else Path(folder)


def get_number_input(
    number_type=int,
    *,
    prefix: str = "What number do you want",
    allow_none: bool = False,
    default_val=None,
    min_val=None,
    max_val=None,
):
    """
    A helper function to ask the user for a number value.
    Also does some validation and allows for default input etc.

    :param number_type: The type of number to return. Use python's built in ``int``
        or ``float`` functions.
    :param prefix: The question to ask the user.
    :param allow_none: Is None a valid return type?
    :param default_val: Optional parameter to provide if ``allow_none`` is ``True``.
        Ignored if ``allow_none`` is ``False``.
    :param min_val: The smallest allowable value (inclusive). If None, this is ignored.
    :param max_val: The largest allowable value (inclusive). If None, this is ignored.
    :return: A number, or if allowed, None.
    """

    number = None
    ok = False
    prefix = prefix.strip()

    input_prompt = _input_prompt(
        allow_none, default_val, max_val, min_val, number_type, prefix
    )

    while not ok:
        number = input(input_prompt)

        if number is None and allow_none:
            break

        if ((number == "") or (number.lower() == "none")) and allow_none:
            number = default_val
            break

        with contextlib.suppress(ValueError):
            number = number_type(number)

            if min_val is None and max_val is None:
                break

            if max_val is None:
                if number >= min_val:
                    break
                continue
            if min_val is None:
                if number <= max_val:
                    break
                continue
            if number >= min_val and number <= max_val:
                break

    return number


def _input_prompt(allow_none, default_val, max_val, min_val, number_type, prefix):
    """
    Helper method to build the input prompt for the get_number_input function.

    :param allow_none: Are None values allowed?
    :param default_val: Default value to use if None entered.
    :param max_val: The maximum value.
    :param min_val: The minimum value.
    :param number_type: The type of number expected (float, int).
    :param prefix: The prefix to put before the prompt.
    """

    allow_none_string = ""

    if allow_none:
        if default_val is None:
            default_on_none = "None"
        elif isinstance(default_val, number_type):
            default_on_none = f"Default={default_val}"

        else:
            raise ValueError(f"Default value is not the same type as {number_type}")

        allow_none_string = " (Use 'None', 'none' or '' for " + f"{default_on_none})"

    if min_val is None and max_val is None:
        limit_string = ""
    elif max_val is None:
        limit_string = f", number must be >={min_val}"
    elif min_val is None:
        limit_string = f", number must be <={max_val}"
    else:
        limit_string = f", number must satisfy {min_val}<=number<={max_val}"

    return prefix + limit_string + allow_none_string + ": "
