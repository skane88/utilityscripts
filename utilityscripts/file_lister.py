"""
Lists all the folders / files in a given directory, and also has the ability
to save the list to a text file.
"""

from pathlib import Path

from ui_funcs import get_folder, get_true_false


def main():
    """
    Lists all the folders / files in a given directory, and also has the ability
    to save the list to a text file.
    """

    base_path = get_folder(save=False, type_prompt="Folder to Search")

    print()

    if not base_path.is_dir():
        print("The provided path is not a directory.")
        print()
        input("Press any key to exit.")
        return

    file_lister(base_path)

    print()
    input("Press any key to exit.")


def file_lister(base_path: Path):
    """
    list the files given a base path to search.

    :param base_path: The path to search.
    """

    filter_val = "*"
    different_folder: bool = None
    recursive = get_true_false(prefix="Do you want to search subfolders")
    report_folders = get_true_false(prefix="Do you want to report folders")
    incl_full_path = get_true_false(prefix="Do you want to include the full path")

    incl_relative_path = (
        False if incl_full_path else get_true_false(prefix="Relative paths")
    )

    incl_extension = get_true_false(prefix="Do you want to include the extension")

    if get_true_false(prefix="Do you want to filter by file type"):
        filter_val = input(
            "Input a valid 'glob' type filter (i.e. '*', '*.' or '*.pdf'): "
        )

    save_to_file = get_true_false(prefix="Do you want to save to a file")
    if save_to_file:
        different_folder = get_true_false(
            prefix="Do you want to save the list to a different location?",
            allow_quit=False,
        )
    if different_folder:
        save_folder = get_folder(type_prompt="Output Folder")
    else:
        save_folder = base_path
    all_lower_case = get_true_false(
        prefix="Do you want to convert file names to lower case"
    )
    sort_file = get_true_false(prefix="Do you want to sort the file names")
    # now we've got input, now do the actual finding of files
    if recursive:
        # if subfolders are required, use rglob
        f_iterator = base_path.rglob(filter_val)
    else:
        # if only the local folder, just use iterdir
        f_iterator = base_path.glob(filter_val)
    print()
    text_to_save = []
    for file in f_iterator:
        file: Path

        if not report_folders and file.is_dir():
            continue

        text = ""

        if incl_full_path:
            text = str(file)
        else:
            text = str(file.relative_to(base_path)) if incl_relative_path else file.name

        if not incl_extension:
            text = text.replace(file.suffix, "")

        if save_to_file:
            text_to_save += [text]
        else:
            print(text)
    # now save to file if necessary
    if save_to_file:
        # lower case everything if required
        if all_lower_case:
            text_to_save = [t.lower() for t in text_to_save]

        # sort list of files if required
        if sort_file:
            text_to_save = sorted(text_to_save)

        # first get a file name
        start_int = 1

        file_ext = ".txt"
        file_name = "FileLister "
        file_exists = True

        while file_exists:
            output_file = save_folder / Path(f"{file_name}{start_int:02d}").with_suffix(
                file_ext
            )

            file_exists = output_file.exists()
            start_int += 1

        with open(output_file, "w") as file:
            for line in text_to_save:
                file.write(f"{line}\n")


if __name__ == "__main__":
    main()
