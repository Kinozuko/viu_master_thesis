from PIL import Image

def split_image(image_path, output_folder, num_splits):
    img = Image.open(image_path)
    
    width, height = img.size
    
    split_height = height // num_splits
    
    for i in range(num_splits):
        # Define the bounding box for the current split
        left = 0
        top = i * split_height
        right = width
        bottom = (i + 1) * split_height if i != num_splits - 1 else height
        
        img_cropped = img.crop((left, top, right, bottom))
        
        img_cropped.save(f"{output_folder}/image_part_{i + 1}.png")
        
    print(f"Image split into {num_splits} parts and saved in {output_folder}")

image_path = "models/v2/model_v2.png"
output_folder = "models/v2/splitted_image"
num_splits = 4 

split_image(image_path, output_folder, num_splits)
