import os
import nibabel as nib
from collections import defaultdict

def count_files_by_dimension(root_folder, new_dataset_folder):
    """
    Count the number of files in each dimensionality (2D, 3D, 4D, etc.) for the given dataset folders.

    :param root_folder: Path to the root folder containing all patient folders.
    :param new_dataset_folder: Path to the new dataset folder where files will be checked.
    :return: None
    """
    dimension_type_counts = defaultdict(int)

    base_dir = os.path.join(root_folder, new_dataset_folder)
    folders = ['Flair', 'T1', 'T2']

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                file_path = os.path.join(folder_path, file)
                # Load the NIfTI image
                img = nib.load(file_path)
                img_shape = img.shape
                # Determine the dimensionality (2D, 3D, 4D, etc.)
                dimension_type = len(img_shape)
                # Increment the count for the corresponding dimension type
                dimension_type_counts[dimension_type] += 1

    for dimension_type, count in sorted(dimension_type_counts.items()):
        print(f"Files with {dimension_type}D -> {count}")

if __name__ == '__main__':
    root_folder = 'datasets'
    new_dataset_folder = 'new_dataset'

    count_files_by_dimension(root_folder, new_dataset_folder)
