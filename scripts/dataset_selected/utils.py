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