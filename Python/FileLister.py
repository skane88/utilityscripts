import sys
from pathlib import Path


def lister():
    base_path = Path(input("Provide the base path: "))

    print()

    if base_path.is_dir():

        true_dict = {"n": False, "no": False, "y": True, "yes": True}
        recursive: bool = None
        report_folders: bool = None
        incl_full_path: bool = None
        incl_extension: bool = None
        filter_type: bool = None
        filter_val = "*"

        while recursive is None:
            r = input("Do you want to search subfolders (Y or N)? ")

            recursive = true_dict.get(r.lower(), None)

        while report_folders is None:
            r = input("Do you want to report folders (Y or N)? ")

            report_folders = true_dict.get(r.lower(), None)

        while incl_full_path is None:
            fp = input("Do you want to include the full file path (Y or N)? ")

            incl_full_path = true_dict.get(fp.lower(), None)

        while incl_extension is None:
            ie = input("Do you want to include the extension (Y or N)? ")

            incl_extension = true_dict.get(ie.lower(), None)

        while filter_type is None:
            ft = input("Do you want to filter by file type (Y or N)? ")

            filter_type = true_dict.get(ft.lower(), None)

        if filter_type:
            filter_val = input(
                "Input a valid 'glob' type filter (i.e. '*', '*.' or '*.pdf'): "
            )

        # now we've got input, now do the actual finding of files

        if recursive:
            # if subfolders are required, use rglob
            f_iterator = base_path.rglob(filter_val)
        else:
            # if only the local folder, just use iterdir
            f_iterator = base_path.glob(filter_val)

        print()

        for f in f_iterator:

            f: Path()

            if not report_folders:
                if f.is_dir():
                    continue

            text = str(f)

            if not incl_full_path:
                text = f.name

            if not incl_extension:
                text = text.replace(f.suffix, "")

            print(text)

    else:
        print("The provided path is not a directory.")

    print()
    input("Press any key to exit.")


if __name__ == "__main__":
    lister()
