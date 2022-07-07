"""
I believe that this sorts out photos etc. with an Action No. in their name into
appropriate folders...but it's been a while and I didn't comment this properly at the
time, so who knows?
"""

import sys
from tqdm import tqdm
from pathlib import Path


def re_name():

    true_dict = {"n": False, "no": False, "y": True, "yes": True}

    print()
    print("This will rename the folders, and cannot be reversed.")

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

        filter_val = "*"

        # now we've got input, now do the actual finding of files
        f_iterator = base_path.rglob(filter_val)

        print()

        changed = 0

        for f in tqdm(f_iterator, desc="Renaming Folders", unit="Folders"):

            f: Path()

            if f.is_dir():
                continue

            folder_name = f.parent.name
            folder_parent = f.parent.parent
            file_name = f.name

            folder_action_no = folder_name.split(" ")[1]

            new_path = folder_parent / folder_action_no / file_name

            if new_path.exists():

                warning_string = "ERROR\n"
                warning_string += f"Tried replacing: \n    {f}\nwith\n    {new_path}\n"
                warning_string += f"But new path already exists\n"
                warning_string += "-" * 40
                warning_string += "\n"

                warnings += [warning_string]

                continue

            if not new_path.parent.exists():
                new_path.parent.mkdir(parents=True)

            f.rename(new_path)
            changed += 1

        print()
        print(f"Updated {changed} files.")

    else:
        print("The provided path is not a directory.")

    print()
    input("Press any key to exit.")


if __name__ == "__main__":
    re_name()
