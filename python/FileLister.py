"""
Lists all the folders / files in a given directory, and also has the ability to save
the list to a text file.
"""

from pathlib import Path


def lister():
    base_path = Path(input("Provide the base path: "))

    print()

    if base_path.is_dir():

        true_dict = {"n": False, "no": False, "y": True, "yes": True}
        recursive: bool = None
        report_folders: bool = None
        incl_full_path: bool = None
        incl_relative_path: bool = None
        incl_extension: bool = None
        filter_type: bool = None
        filter_val = "*"
        save_to_file: bool = None

        while recursive is None:
            r = input("Do you want to search subfolders (Y or N)? ")

            recursive = true_dict.get(r.lower())

        while report_folders is None:
            r = input("Do you want to report folders (Y or N)? ")

            report_folders = true_dict.get(r.lower())

        while incl_full_path is None:
            fp = input("Do you want to include the full file path (Y or N)? ")

            incl_full_path = true_dict.get(fp.lower())

        if not incl_full_path:
            while incl_relative_path is None:
                rp = input("Do you want to include the relative path (Y or N)?")

                incl_relative_path = true_dict.get(rp.lower())
        else:
            incl_relative_path = False

        while incl_extension is None:
            ie = input("Do you want to include the extension (Y or N)? ")

            incl_extension = true_dict.get(ie.lower())

        while filter_type is None:
            ft = input("Do you want to filter by file type (Y or N)? ")

            filter_type = true_dict.get(ft.lower())

        if filter_type:
            filter_val = input(
                "Input a valid 'glob' type filter (i.e. '*', '*.' or '*.pdf'): "
            )

        while save_to_file is None:
            stf = input("Do you want to save to a file (Y or N)?")

            save_to_file = true_dict.get(stf.lower())

        # now we've got input, now do the actual finding of files

        if recursive:
            # if subfolders are required, use rglob
            f_iterator = base_path.rglob(filter_val)
        else:
            # if only the local folder, just use iterdir
            f_iterator = base_path.glob(filter_val)

        print()

        text_to_save = []

        for f in f_iterator:

            f: Path()

            if not report_folders and f.is_dir():
                continue

            text = ""

            if incl_full_path:
                text = str(f)
            else:
                if incl_relative_path:
                    text = str(f.relative_to(base_path))
                else:
                    text = f.name
            if not incl_extension:
                text = text.replace(f.suffix, "")

            if save_to_file:
                text_to_save += [text]
            else:
                print(text)

        # now save to file if necessary
        if save_to_file:

            # first get a file name
            start_int = 1

            file_ext = ".txt"
            file_name = "FileLister "
            file_exists = True

            output_file: Path

            while file_exists:

                output_file = base_path / Path(
                    f"{file_name}{start_int:02d}"
                ).with_suffix(file_ext)

                file_exists = output_file.exists()
                start_int += 1

            with open(output_file, "w") as f:

                for l in text_to_save:
                    f.write(f"{l}\n")

    else:
        print("The provided path is not a directory.")

    print()
    input("Press any key to exit.")


if __name__ == "__main__":
    lister()
