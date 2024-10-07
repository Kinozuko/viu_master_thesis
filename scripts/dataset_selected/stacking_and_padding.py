import os
import numpy as np
import tensorflow as tf
from PIL import Image

def read_image(filepath):
    img = Image.open(filepath).convert('L')  # Convert to grayscale
    img = np.array(img)
    return img

# Function to stack slices and create 3D volumes
def stack_slices(image_folder):
    volumes = {}
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith('.png'):
            # Extract type, index, and slice number from filename
            parts = filename.split('_')
            img_type, img_index, img_slice = parts[1], parts[2], parts[3].split('.')[0]
            img_index = int(img_index)
            img_slice = int(img_slice)
            
            # Create a new volume key combining img_type and img_index
            volume_key = f"{img_type}_{img_index}"
            
            # Create a new volume if the combination of type and index is new
            if volume_key not in volumes:
                volumes[volume_key] = []
            
            # Read the image and append to the corresponding volume with slice number
            filepath = os.path.join(image_folder, filename)

            img = read_image(filepath)
            volumes[volume_key].append((img_slice, img))  # Store slice number and image

    # Sort slices within each volume by slice number and stack
    for key in volumes:
        # Sort by slice number
        volumes[key] = np.stack([img for _, img in sorted(volumes[key], key=lambda x: x[0])])

    return volumes

def pad_volumes(volumes):
    # Find the maximum number of slices across all volumes
    max_slices = max(vol.shape[0] for vol in volumes.values())
    
    padded_volumes = {}
    
    for key, vol in volumes.items():
        current_slices = vol.shape[0] 
        
        if current_slices == max_slices:
            # No need to pad if the volume already has the max number of slices
            padded_volumes[key] = vol
            continue
        
        # Calculate how many slices need to be added
        missing_slices = max_slices - current_slices
        
        # Proportional duplication: Calculate how many times each slice should be duplicated
        duplication_factor = max_slices // current_slices
        remaining_slices = max_slices % current_slices
        
        new_slices = []
        
        for i in range(current_slices):
            new_slices.append(vol[i])
            
            for _ in range(duplication_factor - 1):
                new_slices.append(vol[i])
            
            if remaining_slices > 0:
                new_slices.append(vol[i])
                remaining_slices -= 1
        
        new_slices = new_slices[:max_slices]
        
        padded_vol = np.stack(new_slices, axis=0)
        padded_volumes[key] = padded_vol
    
    return padded_volumes

# Function to load data using generator
def create_tf_dataset(volumes, batch_size):
    volume_list = [tf.convert_to_tensor(vol, dtype=tf.float32) for vol in volumes.values()]

    dataset = tf.data.Dataset.from_tensor_slices(volume_list)

    dataset = dataset.map(lambda vol: tf.convert_to_tensor(vol, dtype=tf.float32))

    dataset = dataset.batch(batch_size)

    return dataset

def process_images(image_folder, batch_size):
    # Stack the slices into volumes
    volumes = stack_slices(image_folder)
    # Pad the volumes to ensure they have the same dimensions
    padded_volumes = pad_volumes(volumes)
    # Create tensorflow dataset
    dataset = create_tf_dataset(padded_volumes, batch_size=batch_size)
    return dataset

def split_dataset(dataset):
    # Total number of batches in the dataset
    total_batches = dataset.cardinality().numpy()

    train_batches = 4
    val_batches = 1
    test_batches = total_batches - train_batches - val_batches 

    train_dataset = dataset.take(min(train_batches, total_batches))
    
    remaining_dataset = dataset.skip(train_batches)
    
    val_dataset = remaining_dataset.take(min(val_batches, total_batches - train_batches))
    
    remaining_dataset = remaining_dataset.skip(val_batches)
    
    test_dataset = remaining_dataset.take(min(test_batches, total_batches - train_batches - val_batches))

    return train_dataset, val_dataset, test_dataset

# Function to serialize a single example
def serialize_example(volume):
    volume_tensor = tf.io.serialize_tensor(volume) 
    feature = {
        'volume': tf.train.Feature(bytes_list=tf.train.BytesList(value=[volume_tensor.numpy()]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# Function to save dataset to TFRecord
def write_tfrecord(dataset, file_path):
    with tf.io.TFRecordWriter(file_path) as writer:
        for batch in dataset:  # Iterate over batches in the dataset
            for volume in batch:  # Each volume in the batch
                serialized_volume = serialize_example(volume)
                writer.write(serialized_volume)


if __name__ == '__main__':
    image_folder = 'datasets/dataset_selected'
    dataset = process_images(image_folder, batch_size=2)

    train, val, test = split_dataset(dataset)

    print(f"Train dataset cardinality: {train.cardinality().numpy()}")
    print(f"Test dataset cardinality: {val.cardinality().numpy()}")
    print(f"Val dataset cardinality: {test.cardinality().numpy()}")

    write_tfrecord(train, 'datasets/train.tfrecord')
    write_tfrecord(val, 'datasets/val.tfrecord')
    write_tfrecord(test, 'datasets/test.tfrecord')
    
    print(f"Train dataset cardinality: {train.cardinality().numpy()}")
    print(f"Test dataset cardinality: {val.cardinality().numpy()}")
    print(f"Val dataset cardinality: {test.cardinality().numpy()}")