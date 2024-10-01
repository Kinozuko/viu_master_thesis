import os
import pandas as pd

def analyze_mri_images(dataset_path):
    """
    Get number of the unique MRI files stored in the path

    The images are expected to follow this format: image_{type}_{index}_{slice}.png
    where:
    - type: flair, t1, t2 (the type of MRI scan)
    - index: a unique index for each MRI image
    - slice: the slice number of the image

    :param dataset_path: Folder name containing the MRI image slices in PNG format
    :return: DataFrame with columns 'type', 'unique MRI', and 'slices'
    """
    mri_data = {
        'flair': {'unique_indexes': set(), 'total_slices': 0},
        't1': {'unique_indexes': set(), 'total_slices': 0},
        't2': {'unique_indexes': set(), 'total_slices': 0}
    }

    for filename in os.listdir(dataset_path):
        if filename.endswith('.png'):
            parts = filename.split('_')  # Split filename by '_'
            if len(parts) == 4:
                image_type = parts[1]  # flair, t1, or t2
                image_index = parts[2]  # index of the MRI image

                # Add the index to the dictionary based on type
                if image_type in mri_data:
                    mri_data[image_type]['unique_indexes'].add(image_index)
                    mri_data[image_type]['total_slices'] += 1

    data = {
        'type': list(mri_data.keys()),
        'unique MRI': [len(mri_data[image_type]['unique_indexes']) for image_type in mri_data],
        'slices': [mri_data[image_type]['total_slices'] for image_type in mri_data]
    }

    return pd.DataFrame(data)

if __name__ == '__main__':
    dataset_path = 'datasets/dataset_selected'
    df = analyze_mri_images(dataset_path)
    print(df)
