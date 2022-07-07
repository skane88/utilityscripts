"""
This file renumbers photos in a folder to integer numbers starting at 0001.
"""

from pathlib import Path

from tqdm import tqdm


def first_num(filename: Path):
    """
    Get a number to sort filenames based on.
    If there are spaces, it is clearly not a numeric file name.
    If the first character in the name is not a digit, then it is not a number.
    If the first character is a number then sort based on that.
    """

    filename = filename.name

    filename = filename.split(" ")

    if len(filename) == 1:
        return 0

    if not filename[0].isdigit():
        return 0

    return int(filename[0])


def rename_photos():
    """
    This file renumbers photos in a folder to integer numbers starting at 0001.
    """

    true_dict = {"n": False, "no": False, "y": True, "yes": True}

    print()
    print("This will rename the files, and cannot be reversed.")

    while True:
        keep_going = input("Do you wish to continue (y or n)? ")

        keep_going = keep_going.lower()

        if keep_going in true_dict:
            keep_going = true_dict[keep_going]
            break

    if not keep_going:
        return

    print()
    base_path = Path(input("Provide the base path: "))

    if base_path.is_dir():

        recursive: bool = None
        print()
        while recursive is None:
            recursive = input("Do you want to search subfolders (Y or N)? ")

            recursive = true_dict.get(recursive.lower())

        file_types = [".jpg", ".png", ".bmp", ".jpeg"]

        # now we've got input, now do the actual finding of files

        directories = set()

        if recursive:
            # if subfolders are required, use rglob

            print()
            print("Finding Sub-folders by Parsing Links")

            print()

            filter_val = "*"

            d_list = [
                f
                for f in tqdm(
                    base_path.rglob(filter_val), desc="Parsing links", unit="Links"
                )
                if f.is_dir()
            ]

            for directory in d_list:
                directories.add(directory)

        else:
            # if only the local folder, just use iterdir

            directories.add(base_path)

        print()

        changed = 0

        warnings = []

        for folder in tqdm(
            sorted(directories), desc="Searching for Photos", unit="Files"
        ):

            counter = 1

            folder: Path

            # now go through the files / photos etc.
            for photo in tqdm(
                sorted(folder.iterdir(), key=first_num),
                desc="Renaming Photos",
                unit="Photos",
            ):

                photo: Path

                if photo.is_dir():
                    continue

                if photo.suffix.lower() in file_types:
                    # now we need to rename
                    new_path = photo.parent / Path(f"{counter:04d}").with_suffix(
                        photo.suffix
                    )

                    if new_path.exists():

                        warning_string = "ERROR\n"
                        warning_string += (
                            f"Tried replacing: \n    {folder}\n"
                            + f"with\n    {new_path}\n"
                        )
                        warning_string += "But new path already exists\n"
                        warning_string += "-" * 40
                        warning_string += "\n"

                        warnings += [warning_string]

                        continue

                    photo.rename(new_path)
                    changed += 1

                    counter += 1

        print()
        print(f"Updated {changed} files")

    else:
        print("The provided path is not a directory.")

    print()
    input("Press any key to exit.")


if __name__ == "__main__":
    rename_photos()
