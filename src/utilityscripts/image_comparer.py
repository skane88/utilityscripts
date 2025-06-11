"""
A script to quickly compare 2x images using PILlow
"""

from pathlib import Path
from tkinter import Tk, filedialog
from typing import List, Tuple

import imagehash  # type: ignore
from imagehash import ImageHash  # type: ignore
from PIL import Image, ImageFile

from utilityscripts.ui_funcs import get_number_input

DIFFERENCE_THRESHOLD = 20  # how similar should images be?


def get_file_to_open(
    *, type_prompt: str, filetypes: List[Tuple[str, str]] | None = None
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


def get_hash(*, image: Path, hash_size: int = 8) -> ImageHash:
    """
    Get the hash of an image
    :param image: Path to Image.
    :param hash_size: The hash size to use.
        Higher numbers preserve more information about the image when the
        comparison is done. Recommend it is not set below 8.
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

    hash_1 = get_hash(image=image_1, hash_size=hash_size)
    hash_2 = get_hash(image=image_2, hash_size=hash_size)

    return abs(hash_1 - hash_2)


def main():
    """
    Compare 2x images to see if they are the same.
    """

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

    hash_size = get_number_input(
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
        print(
            "Images are identical at the given hash size.\n"
            + "This may change at larger hash sizes."
        )

    else:
        print(
            "NOTE: This is a meaningless number that depends on the hash size used.\n"
            + "Smaller numbers indicate more similar images.\n"
            + "How close images should be is up to the user."
        )


if __name__ == "__main__":
    main()
