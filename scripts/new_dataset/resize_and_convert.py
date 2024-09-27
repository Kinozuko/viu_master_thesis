import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

def convert_nii_to_png(input_dir, output_dir='datasets/final_dataset', target_size=(128, 128)):
    """
    Converts .nii images into PNG slices, stores them in a single folder with a specified naming convention,
    and resizes the images for further processing.
    
    Args:
        input_dir (str): The input directory containing subfolders Flair, T1, T2 with .nii files.
        output_dir (str): The directory where the PNG images will be stored (default is 'final_dataset').
        target_size (tuple): The target size for resizing the images (default is (128, 128).
        
    Returns:
        None
    """
    
    subfolders = ['Flair', 'T1', 'T2']
    output_subfolders = ['flair', 't1', 't2']
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subfolder, output_subfolder in zip(subfolders, output_subfolders):
        print(f"Processing {subfolder} images")
        folder_path = os.path.join(input_dir, subfolder)
        
        nii_file_number = 1  # Keeps track of the .nii file number (x in the naming format)
        
        for file in os.listdir(folder_path):
            if file.endswith('.nii'):
                print(f"Processing slides from nii image #{nii_file_number}")
                nii_path = os.path.join(folder_path, file)
                img = nib.load(nii_path)
                img_data = img.get_fdata()
                
                # Loop through the slices (y in the naming format)
                for slice_idx in range(img_data.shape[2]):
                    slice_img = img_data[:, :, slice_idx]
                    
                    # Normalize the slice
                    slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
                    slice_img = (slice_img * 255).astype(np.uint8)
                    
                    # Resize the image
                    slice_img_resized = resize(slice_img, target_size, anti_aliasing=True)
                    
                    # Format image_type_x_y.png
                    output_filename = f"image_{output_subfolder}_{nii_file_number}_{slice_idx + 1}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    plt.imsave(output_path, slice_img_resized, cmap='gray')
                
                nii_file_number += 1 
    print("Processing Done")
    
if __name__ == '__main__':
    input_directory = 'datasets/new_dataset'
    convert_nii_to_png(input_directory, target_size=(128, 128))
