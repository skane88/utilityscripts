"""
Contains a utility for resizing images before they are saved.
"""

import io
import math
import multiprocessing
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image, ImageFile, UnidentifiedImageError
from pillow_heif import register_heif_opener
from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn
from rich.prompt import Confirm

from utilityscripts.ui_funcs import get_folder, get_number_input

register_heif_opener()

VALID_EXTENSIONS = [".jpg", ".jpeg", ".bmp", ".png", ".heic"]


class ImageResizeError(Exception):
    """Exception raised when errors occur in this module"""


def _validate_file(*, file_path: Path, missing_ok: bool):
    """
    Helper function to validate an image.

    :param file_path: The file path to the image.
    :param missing_ok: Is it ok for an image to be missing,
        or should the function raise an error.
    :return: True if the image exists and is of the correct extension.
        Otherwise, False. Raises an error if it is not OK for the image
        to be missing.
    """
    if not file_path.exists:
        if not missing_ok:
            raise FileNotFoundError(f"Could not find file at {file_path}")

        return False

    if file_path.suffix.lower() not in VALID_EXTENSIONS:
        return False

    return True


def _open_image(file_path: Path) -> Image:
    """
    Helper function to open an image and retry on one cause of failure.
    :param file_path: The image to open.
    """

    def open_jpeg(file_path: Path) -> Image:
        """
        Helper function to open as a JPEG.

        :param file_path: The image to open.
        """

        p: Image = Image.open(file_path)

        if p.format != "JPEG":
            p = p.convert("RGB")

        return p

    try:
        image = open_jpeg(file_path=file_path)

    except OSError:
        # one cause of potential errors is that a jpg file is truncated.
        # therefore try allowing PILlow to open truncated images.
        # we could leave this on be default (set .LOAD_TRUNCATED_IMAGES = True
        # in the import section of the module) but it appears to dramatically
        # slow down PILlow so we will try and leave it as False where possible.

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        print(f"Warning: image {file_path} is a truncated image.")
        image = open_jpeg(file_path=file_path)

        ImageFile.LOAD_TRUNCATED_IMAGES = False

    return image


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

    if not _validate_file(file_path=file_path):
        return None

    current_size = file_path.stat().st_size  # file size in bytes

    if current_size <= target_size:
        # don't resize if already acceptable

        return file_path

    quality_min_orig = quality_min  # keep a record of the original minimum quality

    image = _open_image(file_path)

    # first do a check that at the minimum compression we will be smaller than the
    # target file size

    min_size = _get_size(picture_to_size=image, quality=quality_min)

    if min_size >= current_size:
        # we have specified not to save if larger than the original, so bail out
        # realistically this should be a very rare event so may not even be testable.
        return file_path

    if min_size >= target_size:
        # if the minimum size based on minimum quality is greater than the target size
        # there is no point doing binary search, just choose minimum size.
        final_quality = quality_min
        size = min_size

    else:
        # else the target size will be achieved with a quality between max and min.
        final_quality = -1

        while quality_min <= quality_max:
            # use binary search to quickly converge on an acceptable level of quality.

            quality_average = math.floor((quality_min + quality_max) / 2)
            size = _get_size(picture_to_size=image, quality=quality_average)

            if size <= target_size:
                final_quality = quality_average
                quality_min = quality_average + 1
            else:
                quality_max = quality_average - 1

        if final_quality <= -1:
            raise ImageResizeError("No valid quality level found")

        size = _get_size(picture_to_size=image, quality=final_quality)

    # now we have figured out the level of quality required, resize the image.

    if final_quality >= quality_min_orig:
        if size > target_size and not save_larger_than_target:
            raise ImageResizeError("No valid quality level found, not saving.")

        # get a new file path in case the old was not a jpg
        new_file_path = file_path.with_suffix(".jpg")

        _save_image(
            image=image,
            file_path=new_file_path,
            quality=final_quality,
            img_format="JPEG",
        )

        # need to delete the original file if not a JPG.
        if delete_orig and new_file_path != file_path and file_path.exists():
            file_path.unlink()

    else:
        # if we get to here, we seem to have an error.
        raise ImageResizeError(
            (
                "Expected quality level to be => than minimum allowable. "
                + f"Values were: calculated quality ={final_quality}, "
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

    _save_image(image=picture_to_size, file_path=buffer, quality=quality)

    size = buffer.getbuffer().nbytes
    buffer.close()

    return size


def _save_image(
    *,
    image: Image,
    file_path: Union[Path, io.BytesIO],
    quality: int,
    img_format: str = "JPEG",
    exif=None,
):
    """
    Save an image file to the provided path, at the provided quality.

    :param image: The image to save.
    :param file_path: The path to save it to.
    :param quality: The quality level to save at.
    :param img_format: The file format to save in. Must be a format known to PILlow.
    :param exif: Any exif data to save to image in a format compatible with PILlow.
        If None, any existing exif data on the image will be used instead.
    """

    if exif is None:
        exif = image.info.get("exif", None)
    try:
        if exif is None:
            image.save(fp=file_path, format=img_format, quality=quality)
        else:
            image.save(fp=file_path, format=img_format, quality=quality, exif=exif)

    except OSError:
        # one cause of potential errors is that a jpg file is truncated.
        # therefore try allowing PILlow to open truncated images.
        # we could leave this on be default (set .LOAD_TRUNCATED_IMAGES = True
        # in the import section of the module) but it appears to dramatically
        # slow down PILlow so we will try and leave it as False where possible.

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if exif is None:
            image.save(fp=file_path, format="JPEG", quality=quality)
        else:
            image.save(fp=file_path, format="JPEG", quality=quality, exif=exif)

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

    with multiprocessing.Pool(processes=cpus) as pool:
        total = len(files_to_resize)

        target_size_list = [target_size] * total
        quality_min_list = [quality_min] * total
        quality_max_list = [quality_max] * total
        save_larger_list = [save_larger_than_target] * total
        delete_orig_list = [delete_orig] * total
        missing_ok_list = [missing_ok] * total
        verbose_list = [verbose] * total
        ignore_errors_list = [ignore_errors] * total

        input_vals = zip(
            files_to_resize,
            target_size_list,
            quality_min_list,
            quality_max_list,
            save_larger_list,
            delete_orig_list,
            missing_ok_list,
            verbose_list,
            ignore_errors_list,
            strict=True,
        )

        if verbose:
            with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            ) as progress:
                progress_bar = progress.add_task("Resizing Images...", total=total)

                for vals in pool.imap_unordered(_help_resize, input_vals):
                    new_files.append(vals)

                    progress.advance(progress_bar, advance=1)

        else:
            new_files = pool.imap_unordered(_help_resize, input_vals)

    for _op, new_path, o_size, f_size, warn in new_files:
        output_files += [new_path]
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

    subfolders = Confirm.ask(prompt="Do you want to resize images in subfolders")

    print()
    max_size = get_number_input(
        int,
        prefix="What maximum file size do you want, in kb",
        allow_none=True,
        min_val=50,
        default_val=512,
    )
    max_size = max_size * 1024

    print()
    min_quality = get_number_input(
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

    if Confirm.ask(prompt="Do you wish to continue"):
        _, warnings = compress_all_in_folder(
            folder=folder,
            incl_subfolders=subfolders,
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
