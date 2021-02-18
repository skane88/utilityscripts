from pathlib import Path
from tqdm import tqdm

base_path = Path(input("Provide the base path: "))

if base_path.is_dir():
    filter_val = "*"

    # now we've got input, now do the actual finding of files
    f_iterator = base_path.glob(filter_val)

    empty_folders = set()

    for f in tqdm(f_iterator, desc="Counting files", unit="Folders"):

        count_files = 0

        if not f.is_dir():
            print()
            print(f"{f} is not a folder, continuing")
            continue

        print()
        print(f"Counting files in {f}")

        for file in f.glob(filter_val):
            count_files += 1

        print(f"    No. files is {count_files}")
        if count_files == 0:
            empty_folders.add(f)

    print()

    if len(empty_folders) > 0:
        print("Empty folders:")
        print(empty_folders)
    else:
        print("No empty folders")

else:
    print("Not a directory.")

print("Finished")
print("=" * 40)
