"""
A script to quickly compare 2x images using PILlow
"""

from typing import List, Tuple

from pathlib import Path
from tkinter import Tk, filedialog

from PIL import Image, ImageFile
import imagehash

DIFFERENCE_THRESHOLD = 20  # how similar should images be?


def get_file_to_open(
    *, type_prompt: str, filetypes: List[Tuple[str, str]] = None
) -> Path:
    """
    Gets a file to open.

    :param save: if True, the strings etc. will be for a save operation, if False for a
        load operation.
    :param type_prompt: A string with the type of file to save / get.
        Used for the user prompt string.

        e.g. "photo" or "excel file" or "report"
    :param filetypes: an optional list of tuples specifying the filetypes to diplay.
        Use None or * to show all filetypes. e.g.
            [
                ("Excel Files", ".xl*"),
                ("All Files", "*"),
            ]
    :return: The chosen folder.
    """

    print()
    print(f"Choose the {type_prompt} to load:")

    root = Tk()
    root.withdraw()

    if filetypes is not None:
        file = filedialog.askopenfilename(filetypes=filetypes)
    else:
        file = filedialog.askopenfilename()

    root.destroy()

    return Path(file)


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
        else:
            if not isinstance(default_val, number_type):
                raise ValueError(f"Default value is not the same type as {number_type}")

            default_on_none = f"Default={default_val}"

        allow_none_string = f" (Use 'None', 'none' or '' for " + f"{default_on_none})"

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
            try:
                number = number_type(number)

                if min_val is None and max_val is None:
                    ok = True
                    continue

                if max_val is None:
                    if number >= min_val:
                        ok = True
                        continue
                    else:
                        continue

                if min_val is None:
                    if number <= max_val:
                        ok = True
                        continue
                    else:
                        continue

                if number >= min_val and number <= max_val:
                    ok = True

            except ValueError:
                pass

    return number


def get_hash(*, image: Path, hash_size: int = 8) -> int:
    """
    Get the hash of an image
    :param image: Path to Image.
    :param hash_size: The hash size to use.
        Higher numbers preserve more information about the image when the comparison is done.
        Recommend it is not set below 8.
    :return: The image hash.
    """

    try:

        i = Image.open(image)

        return imagehash.average_hash(i, hash_size=hash_size)

    except OSError as err:

        print("Hit an OSError, trying again.")
        print(f"i1 = {image}")
        print(err)

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        i = Image.open(image)

        i_hash = imagehash.average_hash(i, hash_size=hash_size)

        ImageFile.LOAD_TRUNCATED_IMAGES = False

    return i_hash


def hash_diff(*, image_1: Path, image_2: Path, hash_size: int = 8) -> int:
    """
    Compare 2x images by using the image hash, as per:

    https://stackoverflow.com/questions/52736154/how-to-check-similarity-of-two-images-that-have-different-pixelization

    :param image_1: Path to Image 1 to compare.
    :param image_2: Path to Image 2 to compare.
    :param hash_size: The hash size to use.
        Higher numbers preserve more information about the image.
        Recommend it is not set below 8.
    :return: Are the images the same?
    """

    hash_1 = get_hash(image=image_1, hash_size=8)
    hash_2 = get_hash(image=image_2, hash_size=8)

    return abs(hash_1 - hash_2)


def main():

    print("Compares images to determine if they are the same.")

    print("Choose first image")

    image_1 = get_file_to_open(
        type_prompt="Image",
        filetypes=[
            ("Common Image Types", "*.jpg *.jpeg *.png *.bmp"),
            ("JPGs", "*.jpg *.jpeg"),
            ("PNGs", "*.png"),
            ("BMPs", "*.bmp"),
            ("All Files", "*"),
        ],
    )

    print("Choose second image")

    image_2 = get_file_to_open(
        type_prompt="Image",
        filetypes=[
            ("JPGs", "*.jpg"),
            ("PNGs", "*.png"),
            ("BMPs", "*.bmp"),
            ("All Files", "*"),
        ],
    )

    hash_size = get_number(
        prefix="Do you wish to choose a different hash size for the comparison",
        default_val=8,
        min_val=8,
        allow_none=True,
    )

    difference = hash_diff(image_1=image_1, image_2=image_2, hash_size=hash_size)

    print("Compared the following images:")
    print(image_1)
    print(image_2)
    print()

    print(f"Difference score between images is: {difference}")

    print()

    if difference == 0:
        print("Images are identical at the given hash size.")

    else:
        print(
            f"NOTE: This is a meaningless number that depends on the hash size used.\n"
            + "Smaller numbers indicate more similar images.\n"
            + "How close images should be is up to the user."
        )


if __name__ == "__main__":

    main()
