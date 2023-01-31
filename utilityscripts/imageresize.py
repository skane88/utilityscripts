"""
Contains a utility for resizing images before they are saved.
"""


import contextlib
import io
import math
import multiprocessing as mp
import os
from pathlib import Path
from tkinter import Tk, filedialog
from typing import List, Optional, Set, Tuple, Union

from PIL import Image, ImageFile, UnidentifiedImageError
from pillow_heif import register_heif_opener
from tqdm import tqdm

register_heif_opener()

VALID_EXTENSIONS = [".jpg", ".jpeg", ".bmp", ".png", ".heic"]

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


def compress_image(
    *,
    file_path: Path,
    target_size: int = 512000,
    quality_min: int = 25,
    quality_max: int = 95,
    save_larger_than_target: bool = False,
    delete_orig: bool = True,
    missing_ok: bool = True,
) -> Optional[Path]:
    """
    Uses the PILlow library to compress an image to below a target file size.
    All images are saved as .jpg files.
    Uses an IO Buffer to minimise writes to disk.
    If it can't reach the target file size, without compromising quality it will either
    raise an error or else save the file at the smallest size possible depending on
    the paramter ``save_larger_than_target``
    Based on the method at:
    https://stackoverflow.com/questions/52259476/how-to-reduce-a-jpeg-size-to-a-desired-size
    :param file_path: The path of the photo to resize.
    :param target_size: The target size in bytes of the image.
        Default of 512000 bytes = 500kb
    :param quality_min: The minimum quality setting for the PILlow save function.
    :param quality_max: The maximum quality setting for the PILlow save function.
    :param save_larger_than_target: Will the program save the file if the projected
        file size is larger than the target?
    :param delete_orig: Delete the original if it has a different file extension.
    :param missing_ok: Is it ok to ignore missing files?
    :returns: The resulting file path.
    """

    if not file_path.exists():
        if missing_ok:
            return None
        else:
            raise FileNotFoundError(f"Could not find file at {file_path}")

    if file_path.suffix.lower() not in VALID_EXTENSIONS:
        # if file path is not a valid image, just bail out.
        return None

    current_size = file_path.stat().st_size  # file size in bytes

    if current_size <= target_size:
        # don't resize if already acceptable

        return file_path

    quality_min_orig = quality_min  # keep a record of the original minimum quality

    def open_image(file_path: Path) -> Image:
        p: Image = Image.open(file_path)

        if p.format != "JPEG":
            p = p.convert("RGB")

        return p

    try:
        picture = open_image(file_path=file_path)

    except OSError as err:
        # one cause of potential errors is that a jpg file is truncated.
        # therefore try allowing PILlow to open truncated images.
        # we could leave this on be default (set .LOAD_TRUNCATED_IMAGES = True
        # in the import section of the module) but it appears to dramatically
        # slow down PILlow so we will try and leave it as False where possible.

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        print(f"Warning: image {file_path} is a truncated image.")
        picture = open_image(file_path=file_path)

        ImageFile.LOAD_TRUNCATED_IMAGES = False

    # first do a check that at the minimum compression we will be smaller than the
    # target file size

    min_size = _get_size(picture_to_size=picture, quality=quality_min)

    if min_size >= current_size:
        # we have specified not to save if larger than the original, so bail out
        # realistically this should be a very rare event so may not even be testable.
        return file_path

    if min_size >= target_size:
        # if the minimum size based on minimum quality is greater than the target size
        # there is no point doing binary search, just choose minimum size.
        quality_acceptable = quality_min
        size = min_size

    else:
        # else the target size will be achieved with a quality between max and min.
        quality_acceptable = -1

        while quality_min <= quality_max:
            # use binary search to quickly converge on an acceptable level of quality.

            quality_average = math.floor((quality_min + quality_max) / 2)
            size = _get_size(picture_to_size=picture, quality=quality_average)

            if size <= target_size:
                quality_acceptable = quality_average
                quality_min = quality_average + 1
            else:
                quality_max = quality_average - 1

        if quality_acceptable <= -1:
            raise Exception("No valid quality level found")

        size = _get_size(picture_to_size=picture, quality=quality_acceptable)

    # now we have figured out the level of quality required, resize the image.

    if quality_acceptable >= quality_min_orig:

        if size > target_size and not save_larger_than_target:
            raise Exception("No valid quality level found, not saving.")

        # get a new file path in case the old was not a jpg
        new_file_path = file_path.with_suffix(".jpg")

        _save_image(
            picture=picture,
            file_path=new_file_path,
            quality=quality_acceptable,
            format="JPEG",
        )

        # need to delete the original file if not a JPG.
        if delete_orig and new_file_path != file_path and file_path.exists():
            file_path.unlink()

    else:
        # if we get to here, we seem to have an error.
        raise Exception(
            (
                f"Expected quality level to be => than minimum allowable. Values were: calculated quality ={quality_acceptable}, "
                + f"minimum allowable = {quality_min_orig}"
            )
        )

    return new_file_path


def _get_size(*, picture_to_size, quality) -> int:
    """
    A helper function to just get a potential image size after resizing by saving
    it into a buffer rather than as an image.
    :param picture_to_size: A PILlow Image object.
    :param quality: The quality to save it at.
    """
    buffer = io.BytesIO()
    # create a new buffer every time to prevent saving into the same buffer

    _save_image(picture=picture_to_size, file_path=buffer, quality=quality)

    size = buffer.getbuffer().nbytes
    buffer.close()

    return size


def _save_image(
    *,
    picture: Image,
    file_path: Union[Path, io.BytesIO],
    quality: int,
    format: str = "JPEG",
    exif=None,
):
    """
    Save an image file to the provided path, at the provided quality.
    :param picture: The image to save.
    :param file_path: The path to save it to.
    :param quality: The quality level to save at.
    :param format: The file format to save in. Must be a format known to PILlow.
    :param exif: Any exif data to save to image in a format compatible with PILlow.
        If None, any existing exif data on the image will be used instead.
    """

    if exif is None:
        exif = picture.info["exif"] if "exif" in picture.info else None
    try:

        if exif is None:
            picture.save(fp=file_path, format=format, quality=quality)
        else:
            picture.save(fp=file_path, format=format, quality=quality, exif=exif)

    except OSError as err:
        # one cause of potential errors is that a jpg file is truncated.
        # therefore try allowing PILlow to open truncated images.
        # we could leave this on be default (set .LOAD_TRUNCATED_IMAGES = True
        # in the import section of the module) but it appears to dramatically
        # slow down PILlow so we will try and leave it as False where possible.

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if exif is None:
            picture.save(fp=file_path, format="JPEG", quality=quality)
        else:
            picture.save(fp=file_path, format="JPEG", quality=quality, exif=exif)

        ImageFile.LOAD_TRUNCATED_IMAGES = False


def _help_resize(input_vals):
    """
    Helper function to allow multi-processing of photos when processing a whole folder
    :param input_vals: Expected values:
        original_path,
        target_size,
        quality_min,
        quality_max,
        save_larger_than_target,
        delete_orig,
        missing_ok,
        verbose,
        ignore_errors
    :return:
    """

    (
        original_path,
        target_size,
        quality_min,
        quality_max,
        save_larger_than_target,
        delete_orig,
        missing_ok,
        verbose,
        ignore_errors,
    ) = input_vals

    warning = None
    original_size = original_path.stat().st_size
    if ignore_errors:
        try:
            new_file_path = compress_image(
                file_path=original_path,
                target_size=target_size,
                quality_min=quality_min,
                quality_max=quality_max,
                save_larger_than_target=save_larger_than_target,
                delete_orig=delete_orig,
                missing_ok=missing_ok,
            )

        except UnidentifiedImageError as err:

            warning = [str(err)]
            new_file_path = None

            if verbose:
                print(err)

    else:
        new_file_path = compress_image(
            file_path=original_path,
            target_size=target_size,
            quality_min=quality_min,
            quality_max=quality_max,
            save_larger_than_target=save_larger_than_target,
            delete_orig=delete_orig,
            missing_ok=missing_ok,
        )

    if new_file_path is None:
        new_file_path = original_path
        final_size = original_size
    else:
        final_size = new_file_path.stat().st_size

    return original_path, new_file_path, original_size, final_size, warning


def compress_all_in_folder(
    *,
    folder: Path,
    incl_subfolders: bool = False,
    with_progress: bool = False,
    target_size: int = 512000,
    quality_min: int = 25,
    quality_max: int = 95,
    save_larger_than_target: bool = False,
    delete_orig: bool = True,
    missing_ok: bool = True,
    verbose: bool = True,
    ignore_errors: bool = True,
) -> Tuple[List[Optional[Path]], List[str]]:
    """

    :param folder: The folder to search.
    :param incl_subfolders: Do you want to resize images in subfolders?
    :param with_progress: Do you want a progress indicator?
    :param target_size: The target size in bytes of the image.
        Default of 512000 bytes = 500kb
    :param quality_min: The minimum quality setting for the PILlow save function.
    :param quality_max: The maximum quality setting for the PILlow save function.
    :param save_larger_than_target: Will the program save the file if the projected
        file size is larger than the target?
    :param delete_orig: Delete the original if it has a different file extension.
    :param missing_ok: bool = True,
    :param verbose: Print status updates?
    :param ignore_errors: Is it OK to ignore certain errors? Currently only the
        ``UnidentifiedImageError`` is ignored.
    :returns: A list of the resulting files and any warnings.
    """

    warnings: List[str] = []
    output_files: List[Path] = []

    f_iterator = folder.rglob("*") if incl_subfolders else folder.glob("*")
    files_to_resize = []

    if verbose:
        print()
        print(f"Finding files to compress in {folder}.")

    for f in f_iterator:
        # to avoid any potential issues about modifying folders / files in a path while
        # we are in it, first just walk the path to find potential candidates for
        # resizing

        if f.is_dir():
            # skip directories
            continue
        if f.suffix.lower() not in VALID_EXTENSIONS:
            # skip any files that are not valid
            continue

        files_to_resize.append(f)

    if verbose:
        print()
        print("Done")
        print()
        print("Compressing files:")

    # now walk the files and resize.
    files_to_resize.sort()

    original_size = 0
    final_size = 0
    new_files = []

    cpus = max(int(os.cpu_count() * 0.5), 1)

    with mp.Pool(processes=cpus) as p:

        total = len(files_to_resize)

        t_size = [target_size] * total
        q_min = [quality_min] * total
        q_max = [quality_max] * total
        save_larger = [save_larger_than_target] * total
        d_orig = [delete_orig] * total
        m_ok = [missing_ok] * total
        v_bose = [verbose] * total
        i_errors = [ignore_errors] * total

        input_vals = zip(
            files_to_resize,
            t_size,
            q_min,
            q_max,
            save_larger,
            d_orig,
            m_ok,
            v_bose,
            i_errors,
        )

        if verbose:
            with tqdm(total=total, unit="Images") as pbar:

                for vals in p.imap_unordered(_help_resize, input_vals):
                    new_files.append(vals)
                    pbar.update()
        else:
            new_files = p.imap_unordered(_help_resize, input_vals)

    for op, np, o_size, f_size, warn in new_files:
        output_files = [np]
        original_size += o_size
        final_size += f_size

        if warn is not None:
            warnings += [warn]

    if verbose:
        if files_to_resize:
            print()

            if original_size > 1024**3:
                divisor = 1024**3
                unit = "Gb"
            elif original_size > 1024**2:
                divisor = 1024**2
                unit = "Mb"
            else:
                divisor = 1024
                unit = "kb"

            print(
                f"Original size: {original_size / divisor:.2f}{unit}, "
                + f"final size: {final_size / divisor:.2f}{unit} "
                + f"({final_size / original_size:.2%})"
            )
        else:
            print("No files found to compress.")

    return output_files, warnings


def main():
    """
    Helper function to allow ``compress_all_in_folder`` to be run on its own.
    """

    print("Specify a folder to resize:")
    print()
    folder = get_folder(save=False, type_prompt="photos")

    if folder is None:
        print()
        print("No folder selected.")
        print()
        print("Have a nice day.")
        print()
        return

    subfolders = get_true_false(prefix="Do you want to resize images in subfolders")

    print()
    max_size = get_number(
        int,
        prefix="What maximum file size do you want, in kb",
        allow_none=True,
        min_val=50,
        default_val=250,
    )
    max_size = max_size * 1024

    print()
    min_quality = get_number(
        int,
        prefix="What minimum quality level do you want",
        min_val=0,
        max_val=50,
        allow_none=True,
        default_val=10,
    )

    subfolder_text = " and subfolders." if subfolders else "."
    print()
    print(
        f"About to resize photos to <= {max_size / 1024:.2f}kb in "
        + f"{folder}{subfolder_text}"
    )

    if compress := get_true_false(prefix="Do you wish to continue"):
        files, warnings = compress_all_in_folder(
            folder=folder,
            incl_subfolders=subfolders,
            with_progress=True,
            target_size=max_size,
            save_larger_than_target=True,
            verbose=True,
            ignore_errors=True,
            quality_min=min_quality,
        )

        if len(warnings) > 0:

            print("=" * 40)
            print(f"THERE ARE {len(warnings)} WARNINGS:")
            print()
            for w in warnings:
                print(w)

            print("=" * 40)

    print()
    input("Have a Nice Day! (press any key to exit)\n")


if __name__ == "__main__":
    main()
