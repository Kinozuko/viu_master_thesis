import tensorflow as tf

def parse_tfrecord_fn(example_proto):
    feature_description = {
        'volume': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    volume = tf.io.parse_tensor(parsed_features['volume'], out_type=tf.float32)
    return volume

# Function to load dataset from TFRecord file
def load_tfrecord(file_path):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    return parsed_dataset

def parse_combined_tfrecord_fn(example_proto, add_channel=False):
    feature_description = {
        'volume': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Parse the volume tensor and the label
    volume = tf.io.parse_tensor(parsed_features['volume'], out_type=tf.float32)
    if add_channel:
        volume = tf.expand_dims(volume, axis=-1)
        volume.set_shape([33, 128, 128, 1])

    label = parsed_features['label']
    return volume, label

# Function to load the combined TFRecord files
def load_combined_tfrecord(file_path, batch_size=2, add_channel=False,shuffle=False,buffer_size=20):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    
    parsed_dataset = raw_dataset.map(lambda x: parse_combined_tfrecord_fn(x, add_channel=add_channel))

    if shuffle:
        parsed_dataset = parsed_dataset.shuffle(buffer_size=buffer_size)

    parsed_dataset = parsed_dataset.batch(batch_size)
    
    return parsed_dataset

def count_records(dataset):
    return sum(1 for _ in dataset)