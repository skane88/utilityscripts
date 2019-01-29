from pathlib import Path

def lister():
    base_path = input("Provide the base path: ")

    base_path = Path(base_path)

    if base_path.is_dir():

        true_dict = {"n": False, "no": False, "y": True, "yes": True}
        file_path_bool = None
        recursive_path = None

        while recursive_path is None:
            recursive_path = input("Do you want to include subfolders (Y or N)?")

            recursive_path = true_dict.get(recursive_path.lower(), None)

        while file_path_bool is None:
            file_path_bool = input("Do you want to include the full file path (Y or N)? ")

            file_path_bool = true_dict.get(file_path_bool.lower(), None)

        if recursive_path:
            f_iterator = base_path.rglob('*')
        else:
            f_iterator = base_path.iterdir()

        for f in f_iterator:

            if not file_path_bool:
                print(f.name)
            else:
                print(f)
    else:
        print('The provided path is not a directory.')
    
    print()
    input('Press any key to exit.')

if __name__ == "__main__":
    lister()
