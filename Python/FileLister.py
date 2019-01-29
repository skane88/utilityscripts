import os


def lister():
    base_path = input("Provide the base path: ")

    true_dict = {"n": False, "no": False, "y": True, "yes": True}
    file_path_bool = None

    while file_path_bool is None:
        file_path_bool = input("Do you want to include the full file path (Y or N)? ")

        file_path_bool = true_dict.get(file_path_bool.lower(), None)

    for path, folders, files in os.walk(base_path):

        for fi in files:

            file_path = os.path.join(path, fi)

            if not file_path_bool:
                print(fi)
            else:
                print(file_path)


if __name__ == "__main__":
    lister()
