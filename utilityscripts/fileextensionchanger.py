"""
This file changes files of one extension type to another extension type.
"""

from pathlib import Path

from rich.prompt import Confirm
from tqdm import tqdm


def re_extension():
    print()
    print("This will rename the extension on files, and cannot be reversed.")

    keep_going = Confirm.ask(prompt="Do you wish to continue")

    if not keep_going:
        return

    print()
    base_path = Path(input("Provide the base path: "))

    warnings = []

    if base_path.is_dir():
        recursive = Confirm.ask(prompt="Do you want to search subfolders")
        filter_val = "*"

        print()
        old_ext = input("Input the starting extension type (e.g. '.jpeg'): ").lower()

        print()
        new_ext = input("Input the extension to change it to (e.g. '.jpg'): ").lower()

        # now we've got input, now do the actual finding of files

        if recursive:
            # if subfolders are required, use rglob
            f_iterator = base_path.rglob(filter_val)
        else:
            # if only the local folder, just use iterdir
            f_iterator = base_path.glob(filter_val)

        print()

        changed = 0

        for f in tqdm(f_iterator, desc="Renaming Extensions", unit="Files"):
            f: Path

            if f.is_dir():
                continue

            if f.suffix.lower() != old_ext:
                continue

            new_path = f.with_suffix(new_ext)

            if new_path.exists():
                warning_string = "ERROR\n"
                warning_string += f"Tried replacing: \n    {f}\nwith\n    {new_path}\n"
                warning_string += "But new path already exists\n"
                warning_string += "-" * 40
                warning_string += "\n"

                warnings += [warning_string]

                continue

            f.rename(new_path)
            changed += 1

        print()
        print(f"Updated {changed} files from {old_ext} to {new_ext}")

    else:
        print("The provided path is not a directory.")

    print()
    input("Press any key to exit.")


if __name__ == "__main__":
    re_extension()
