import os
import shutil

def organize_files_from_all_patients(root_folder, new_dataset_folder):
    """
    Organizes NIfTI files from all patient subfolders into Flair, T1, and T2 subfolders inside new_dataset,
    ignoring files with 'Segmentation' or 'LesionSeg' in their names.
    
    :param root_folder: Path to the root folder containing all patient folders.
    :param new_dataset_folder: Path to the new dataset folder where files will be organized.
    """
    # Define subfolders based on file types
    subfolders = ['Flair', 'T1', 'T2']
    
    # Create new_dataset folder and subfolders if they don't exist
    for subfolder in subfolders:
        os.makedirs(os.path.join(new_dataset_folder, subfolder), exist_ok=True)

    # Loop through all patient folders in the root folder
    for patient_folder in os.listdir(root_folder):
        patient_path = os.path.join(root_folder, patient_folder)

        # Ensure we are processing only directories (patient folders)
        if os.path.isdir(patient_path):
            print(f"Processing {patient_folder}...")
            
            # Loop through files in the patient folder
            for file_name in os.listdir(patient_path):
                # Ignore files with 'Segmentation' or 'LesionSeg' in the name
                if 'LesionSeg' in file_name:
                    continue

                # Determine the file type based on the name and copy to the respective subfolder
                if 'Flair' in file_name:
                    target_folder = os.path.join(new_dataset_folder, 'Flair')
                elif 'T1' in file_name:
                    target_folder = os.path.join(new_dataset_folder, 'T1')
                elif 'T2' in file_name:
                    target_folder = os.path.join(new_dataset_folder, 'T2')
                else:
                    # Skip files that do not match the desired categories
                    print(filename)
                    continue

                # Copy the file to the appropriate subfolder
                source_path = os.path.join(patient_path, file_name)
                target_path = os.path.join(target_folder, file_name)
                shutil.copy(source_path, target_path)
                print(f"Copied {file_name} to {target_folder}")

root_folder = os.path.join("datasets", 
    "Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information")

new_dataset_folder = os.path.join("datasets", "new_dataset")

organize_files_from_all_patients(root_folder, new_dataset_folder)
