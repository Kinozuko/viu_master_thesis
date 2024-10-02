from verify_save import load_tfrecord, visualize_middle_slice_from_splits

if __name__ == '__main__':
    train = load_tfrecord('datasets/train_negative.tfrecord').batch(2)
    val = load_tfrecord('datasets/val_negative.tfrecord').batch(2)
    test = load_tfrecord('datasets/test_negative.tfrecord').batch(2)

    visualize_middle_slice_from_splits(train, val, test)