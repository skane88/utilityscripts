from pathlib import Path


def lister():
    base_path = input("Provide the base path: ")

    base_path = Path(base_path)

    if base_path.is_dir():

        true_dict = {"n": False, "no": False, "y": True, "yes": True}
        file_path_bool: bool = None
        recursive_path: bool = None
        incl_extension: bool = None

        while recursive_path is None:
            r = input("Do you want to include subfolders (Y or N)? ")

            recursive_path = true_dict.get(r.lower(), None)

        while file_path_bool is None:
            fp = input(
                "Do you want to include the full file path (Y or N)? "
            )

            file_path_bool = true_dict.get(fp.lower(), None)

        while incl_extension is None:
            ie = input("Do you want to include the extension (Y or N)? ")

            incl_extension = true_dict.get(ie.lower(), None)

        # now we've got input, now do the actual finding of files

        if recursive_path:
            # if subfolders are required, use rglob
            f_iterator = base_path.rglob("*")
        else:
            # if only the local folder, just use iterdir
            f_iterator = base_path.iterdir()

        print()

        for f in f_iterator:
            
            f: Path()

            text = str(f)

            if not file_path_bool:
                text = f.name

            if not incl_extension:
                text = text.replace(f.suffix, '')

            print(text)
            
    else:
        print("The provided path is not a directory.")

    print()
    input("Press any key to exit.")


if __name__ == "__main__":
    lister()
