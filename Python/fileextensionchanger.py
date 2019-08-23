"""
This file changes files of one extension type to another extension type.
"""

import sys
from tqdm import tqdm
from pathlib import Path


def re_extension():

    true_dict = {"n": False, "no": False, "y": True, "yes": True}   

    print()
    print("This will rename the extension on files, and cannot be reversed.")
    
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

    print()

    warnings =[]

    if base_path.is_dir():

        recursive: bool = None
        filter_val = "*"

        print()
        while recursive is None:
            r = input("Do you want to search subfolders (Y or N)? ")

            recursive = true_dict.get(r.lower(), None)

        print()
        old_ext = input("Input the starting extension type (e.g. '.jpeg'): ")

        print()
        new_ext = input("Input the extension to change it to (e.g. '.jpg'): ")

        # now we've got input, now do the actual finding of files

        if recursive:
            # if subfolders are required, use rglob
            f_iterator = base_path.rglob(filter_val)
        else:
            # if only the local folder, just use iterdir
            f_iterator = base_path.glob(filter_val)

        print()

        for f in tqdm(f_iterator, desc="Renaming Extensions", unit="Files"):

            f: Path()

            if f.is_dir():
                continue

            if f.suffix != old_ext:
                continue

            new_path = f.with_suffix(new_ext)

            if new_path.exists():

                warning_string = "ERROR\n"
                warning_string += f"Tried replacing: \n    {f}\nwith\n    {new_path}\n"
                warning_string += f"But new path already exists\n"
                warning_string += "-" * 40
                warning_string += "\n"
                
                warnings += [warning_string]

                continue

            f.rename(new_path)

        print()

    else:
        print("The provided path is not a directory.")

    print()
    input("Press any key to exit.")


if __name__ == "__main__":
    re_extension()
