import os
from collections import defaultdict

def analyze_dataset_folders(root_folder):
    """
    Analyze the dataset folder so we can check how many uniques file formats
    we have and the number of elements, also we summarize the number of elements by extension
    and a breakdown the subfolder dataset based on their format included

    :root_folder (str) -> String format of the root folder

    """
    extension_count = defaultdict(int)  
    extension_folders = defaultdict(lambda: defaultdict(int))  # To count files per root folder per extension
    subfolder_count = defaultdict(int)  # To count the number of subfolders in each root folder

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            # Extract the file extension
            ext = os.path.splitext(filename)[1]
            if ext:
                # Find the root folder
                root_subfolder = os.path.join(root_folder, os.path.relpath(dirpath, root_folder).split(os.sep)[0])
                extension_count[ext] += 1
                extension_folders[root_subfolder][ext] += 1

        # Count the subfolders in the root folder
        root_subfolder = os.path.join(root_folder, os.path.relpath(dirpath, root_folder).split(os.sep)[0])
        subfolder_count[root_subfolder] = len([d for d in os.listdir(root_subfolder) if os.path.isdir(os.path.join(root_subfolder, d))])

    print(f"Number of unique file formats: {len(extension_count)}")
    total_files = sum(extension_count.values())
    print(f"Total number of elements: {total_files}\n")

    print("File Extensions and Total Counts:")
    for ext, count in extension_count.items():
        print(f"Extension: {ext}, Total Count: {count}")
    
    print("\nDetailed Breakdown by Root Folder:")
    for root_folder, extensions in extension_folders.items():
        print(f"\nRoot Folder: {root_folder}:")
        for ext, count in extensions.items():
            print(f"  Extension {ext}: Count: {count}")
        print(f"  Subfolders: {subfolder_count[root_folder]} subfolders")

if __name__ == "__main__":
    root_folder = "datasets/"
    analyze_dataset_folders(root_folder)
