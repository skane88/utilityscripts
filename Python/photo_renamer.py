"""
This file changes files of one extension type to another extension type.
"""

import sys
from tqdm import tqdm
from pathlib import Path


def first_num(filename: Path):

    filename = filename.name

    filename = filename.split(" ")

    if len(filename) ==1:
        return 0

    if not filename[0].isdigit():
        return 0

    return int(filename[0])


def re_extension():

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

    warnings = []

    if base_path.is_dir():

        recursive: bool = None
        filter_val = "*"

        print()
        while recursive is None:
            r = input("Do you want to search subfolders (Y or N)? ")

            recursive = true_dict.get(r.lower(), None)

        file_types = [".jpg", ".png", ".bmp", ".jpeg"]

        # now we've got input, now do the actual finding of files

        directories = set()

        if recursive:
            # if subfolders are required, use rglob

            print()
            print("Finding Sub-folders by Parsing Links")

            print()

            d_list = []

            for f in tqdm(
                base_path.rglob(filter_val), desc="Parsing links", unit="Links"
            ):

                if f.is_dir():

                    d_list.append(f)

            for d in d_list:
                directories.add(d)

        else:
            # if only the local folder, just use iterdir

            directories.add(base_path)

        print()

        changed = 0

        for f in tqdm(sorted(directories), desc="Searching for Photos", unit="Files"):

            counter = 1

            f: Path()

            # now go through the files / photos etc.
            for p in tqdm(
                sorted(f.iterdir(), key=first_num),
                desc="Renaming Photos",
                unit="Photos",
            ):

                p: Path()

                if p.is_dir():
                    continue

                if p.suffix.lower() in file_types:
                    # now we need to rename
                    new_path = p.parent / Path(f"{counter:04d}").with_suffix(p.suffix)

                    if new_path.exists():

                        warning_string = "ERROR\n"
                        warning_string += (
                            f"Tried replacing: \n    {f}\nwith\n    {new_path}\n"
                        )
                        warning_string += f"But new path already exists\n"
                        warning_string += "-" * 40
                        warning_string += "\n"

                        warnings += [warning_string]

                        continue

                    p.rename(new_path)
                    changed += 1

                    counter += 1

        print()
        print(f"Updated {changed} files")

    else:
        print("The provided path is not a directory.")

    print()
    input("Press any key to exit.")


if __name__ == "__main__":
    re_extension()
