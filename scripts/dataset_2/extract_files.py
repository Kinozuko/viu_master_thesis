import os
import gzip

# Define root folders and subdirectories
root_folder = "datasets"
dataset_2 = os.path.join(root_folder, "shifts_ms_pt2")
best_folder = os.path.join(dataset_2, "shifts_ms_pt2/best/train")
folders_to_keep = ['flair', 't1', 't2']  
new_folder = "datasets/new_dataset"

def extract_gz_file(gz_path, output_path):
    """
    Extracts the contents of a .gz file and saves it to the specified output path.
    
    :param gz_path: Path to the .gz file.
    :param output_path: Path where the extracted .nii file will be saved.
    :return: None
    """
    # Open the .gz file in read-binary mode and extract the content
    with gzip.open(gz_path, 'rb') as f_in:
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
    for image_folder in folders_to_keep:
        print(f"Processing folder {image_folder}")
        
        move_to = os.path.join(new_folder, image_folder.capitalize())
        
        source_folder = os.path.join(best_folder, image_folder)
        
        for image in os.listdir(source_folder):
            if image.endswith('.gz'):
                image_path = os.path.join(source_folder, image)
                
                nii_file_name = os.path.splitext(image)[0]
                
                output_path = os.path.join(move_to, nii_file_name)
                
                # Extract the .gz file and save the result in the destination folder
                extract_gz_file(image_path, output_path)
                print(f"Extracted {image} to {output_path}")

if __name__ == '__main__':
    extract_and_move()
