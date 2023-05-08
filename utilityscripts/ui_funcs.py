"""
File to contain basic UI functions.
"""
import contextlib
from pathlib import Path
from tkinter import Tk, filedialog
from typing import Optional, Set

yes_set = {"y", "Y", "Yes", "YES", "yes"}
no_set = {"n", "N", "No", "NO", "no"}
quit_set = {"quit", "q", "Q", "Quit"}


class QuitToMain(Exception):
    """
    An exception to force the program to quit to the main menu.
    """

    pass


def get_true_false(
    *,
    prefix: str = "Enter a yes or no",
    true_false_prompt: str = None,
    true_set: Set[str] = None,
    false_set: Set[str] = None,
    allow_quit: bool = False,
    exit_set: Set[str] = None,
) -> bool:
    """
    Gets a true or false answer from the user.
    Note that it is case insensitive - lower and upper characters are treated alike.
    :param prefix: The message to ask the user about.
    :param true_false_prompt: An optional prompt to describe yes or no. No need to
        provide brackets or question marks.
        For example: "y or n" or "Yes=y, No=n"
        If None, uses the lower case versions of the yes & no chars.
    :param true_set: The characters / words for Yes.
        If None, use the global no_set variable.
    :param false_set: The characters / words for No.
        If None, use the global yes_set variable.
    :param allow_quit: If TRUE, the user can quit by typing quit.
    :param exit_set: the characters / words that cause the program to quit.
    :return:
    """

    prefix = prefix.strip()

    if true_set is None:
        true_set = yes_set

    # make sure there are lower case versions of each char in the set.
    intermediate_set = set()

    for c in true_set:
        intermediate_set.add(c)
        intermediate_set.add(c.lower())

    true_set = intermediate_set

    if false_set is None:
        false_set = no_set

    # make sure there are lower case versions of each char in the set.
    intermediate_set = set()

    for c in false_set:
        intermediate_set.add(c)
        intermediate_set.add(c.lower())

    false_set = intermediate_set

    if allow_quit:
        if exit_set is None:
            exit_set = quit_set

        for e in exit_set:
            if e in true_set:
                raise ValueError(f"quit keyword {e} was in the set of True responses.")
            if e in false_set:
                raise ValueError(f"quit keyword {e} was in the set of False responses.")

    # next build a prompt for the user if none is provided.
    if true_false_prompt is None:
        prompt_set = {c.lower() for c in true_set}

        true_prompt = sorted(list(prompt_set))

        prompt_set = {c.lower() for c in false_set}

        false_prompt = sorted(list(prompt_set))

        if len(true_prompt) == 1:
            true_false_prompt = f"Yes={true_prompt[0]}"
        else:
            true_false_prompt = f"Yes={true_prompt}"

        true_false_prompt += ", "

        if len(false_prompt) == 1:
            true_false_prompt += f"No={false_prompt[0]}"
        else:
            true_false_prompt += f"No={false_prompt}"
    else:
        true_false_prompt = true_false_prompt.strip()
        true_false_prompt = true_false_prompt.lower()

    if allow_quit:
        exit_set = sorted(list(exit_set))

        quit_prompt = f"{exit_set[0]}" if len(exit_set) == 1 else f"{exit_set}"

        question = f"{prefix} ({true_false_prompt}" + f" or {quit_prompt} to quit)? "
    else:
        question = f"{prefix} ({true_false_prompt})? "

    while True:
        print()
        yes_no = input(question)

        if yes_no in true_set:
            return True
        if yes_no in false_set:
            return False
        if allow_quit and yes_no in exit_set:
            raise QuitToMain()


def get_folder(*, save: bool = True, type_prompt: str) -> Optional[Path]:
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


def get_number(
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

    prefix = prefix.strip()
    number = None
    ok = False

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

    while not ok:
        number = input(prefix + limit_string + allow_none_string + ": ")

        if number is None:
            if allow_none:
                ok = True
        elif number == "" or number.lower() == "none":
            number = default_val

            if allow_none:
                ok = True

        else:
            with contextlib.suppress(ValueError):
                number = number_type(number)

                if min_val is None and max_val is None:
                    ok = True
                    continue

                if max_val is None:
                    if number >= min_val:
                        ok = True
                    continue
                if min_val is None:
                    if number <= max_val:
                        ok = True
                    continue
                if number >= min_val and number <= max_val:
                    ok = True

    return number
