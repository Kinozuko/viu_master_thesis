import tensorflow as tf
from utils import parse_tfrecord_fn, count_records

def load_tfrecord_with_label(file_path):
    label = 0 if 'negative' in file_path else 1  
    
    raw_dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    
    # Add label to the volume
    labeled_dataset = parsed_dataset.map(lambda volume: (volume, label))  
    return labeled_dataset

def combine_datasets(positive_file, negative_file, buffer_size=8, batch_size=2):
    # Load datasets with labels
    positive_dataset = load_tfrecord_with_label(positive_file)
    negative_dataset = load_tfrecord_with_label(negative_file)

    # Combine the positive and negative datasets
    combined_dataset = positive_dataset.concatenate(negative_dataset)
    
    # Shuffle and batch the dataset
    combined_dataset = combined_dataset.batch(batch_size)
    return combined_dataset

def serialize_example(volume, label):
    volume_tensor = tf.io.serialize_tensor(volume)
    label_int = int(label)
    feature = {
        'volume': tf.train.Feature(bytes_list=tf.train.BytesList(value=[volume_tensor.numpy()])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_int]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_combined_tfrecord(dataset, file_path):
    with tf.io.TFRecordWriter(file_path) as writer:
        for batch_volumes, batch_labels in dataset: 
            for volume, label in zip(batch_volumes, batch_labels):
                serialized_example = serialize_example(volume, label)
                writer.write(serialized_example)

if __name__ == '__main__':

    train_dataset_combined = combine_datasets(
        'datasets/train.tfrecord',
        'datasets/train_negative.tfrecord')
    val_dataset_combined = combine_datasets(
        'datasets/val.tfrecord',
        'datasets/val_negative.tfrecord')
    test_dataset_combined = combine_datasets(
        'datasets/test.tfrecord',
        'datasets/test_negative.tfrecord')

    write_combined_tfrecord(
        train_dataset_combined,
        'data/final_train.tfrecord')
    write_combined_tfrecord(
        val_dataset_combined,
        'data/final_val.tfrecord')
    write_combined_tfrecord(
        test_dataset_combined, 
        'data/final_test.tfrecord')

    print("Final datasets saved as TFRecord files.")