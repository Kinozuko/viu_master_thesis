import os
import gzip

# Define root folders and subdirectories
root_folder = "datasets"
dataset_2 = os.path.join(root_folder, "shifts_ms_pt2")
best_folder = os.path.join(dataset_2, "shifts_ms_pt2/best/train")
folders_to_keep = ['flair', 't1', 't2']  # Folders that contain specific types of images
new_folder = "datasets/new_dataset"  # Folder to store the extracted files

def extract_gz_file(gz_path, output_path):
    """
    Extracts the contents of a .gz file and saves it to the specified output path.
    
    :param gz_path: Path to the .gz file.
    :param output_path: Path where the extracted .nii file will be saved.
    :return: None
    """
    # Open the .gz file in read-binary mode and extract the content
    with gzip.open(gz_path, 'rb') as f_in:
        # Open the output file in write-binary mode and save the decompressed content
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())

def extract_and_move():
    """
    Loops through the specified image folders, extracts the .gz files,
    and moves the extracted .nii files to a new folder.

    :param root_folder: Path to the root folder containing all patient folders.
    :param new_dataset_folder: Path to the new dataset folder where files will be organized.
    :return: None
    """
    # Iterate over the folders (e.g., 'flair', 't1', 't2')
    for image_folder in folders_to_keep:
        print(f"Processing folder {image_folder}")
        
        # Define the destination folder where extracted files will be moved
        move_to = os.path.join(new_folder, image_folder.capitalize())
        
        # Get the source folder path where .gz files are located
        source_folder = os.path.join(best_folder, image_folder)
        
        # Loop over the files in the source folder
        for image in os.listdir(source_folder):
            # Only process .gz files
            if image.endswith('.gz'):
                # Construct the full path to the .gz file
                image_path = os.path.join(source_folder, image)
                
                # Generate the output file name by removing the .gz extension
                nii_file_name = os.path.splitext(image)[0]
                
                # Define the destination path for the extracted .nii file
                output_path = os.path.join(move_to, nii_file_name)
                
                # Extract the .gz file and save the result in the destination folder
                extract_gz_file(image_path, output_path)
                print(f"Extracted {image} to {output_path}")

if __name__ == '__main__':
    # Run the extraction and file moving process
    extract_and_move()
