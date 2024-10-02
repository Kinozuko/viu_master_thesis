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

# Function to pad volumes to the same shape
def pad_volumes(volumes):
    # Find the maximum shape across all volumes
    max_shape = tuple(np.max([v.shape for v in volumes.values()], axis=0))
    
    padded_volumes = {}
    for key, vol in volumes.items():
        # Calculate padding needed for each dimension
        pad_width = [(0, max_dim - vol_dim) for vol_dim, max_dim in zip(vol.shape, max_shape)]
        padded_vol = np.pad(vol, pad_width, mode='constant', constant_values=0)
        padded_volumes[key] = padded_vol
    
    return padded_volumes

# Function to load data using generator
def create_tf_dataset(volumes, batch_size):
    def generator():
        for key, vol in volumes.items():
            yield vol

    dataset = tf.data.Dataset.from_generator(generator, output_signature=tf.TensorSpec(shape=(None, None, None), dtype=tf.float32))
    dataset = dataset.batch(batch_size)
    return dataset

def process_images(image_folder, batch_size):
    # Stack the slices into volumes
    volumes = stack_slices(image_folder)
    
    # Pad the volumes to ensure they have the same dimensions
    padded_volumes = pad_volumes(volumes)
    
    # Create dataset with generator
    dataset = create_tf_dataset(padded_volumes, batch_size=batch_size)
    
    return dataset

def split_dataset(dataset):
    # Take the first 4 batches for training (8 volumes total)
    train_dataset = dataset.take(4)

    # Skip the first 4 batches (8 volumes), leaving the remaining 3 batches
    remaining_dataset = dataset.skip(4)

    # Take 1 batch for validation (2 volumes)
    val_dataset = remaining_dataset.take(1)

    # The rest (2 batches, 3 volumes) for testing
    test_dataset = remaining_dataset.skip(1)

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

    write_tfrecord(train, 'datasets/train.tfrecord')
    write_tfrecord(val, 'datasets/val.tfrecord')
    write_tfrecord(test, 'datasets/test.tfrecord')

    print("Datasets saved as TFRecord files in datasets folder")