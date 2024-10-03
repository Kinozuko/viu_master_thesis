from stacking_and_padding import process_images, split_dataset, write_tfrecord

if __name__ == '__main__':
    image_folder = 'datasets/dataset_selected_negative'
    dataset = process_images(image_folder, batch_size=2)
    
    train, val, test = split_dataset(dataset)

    print(f"Train dataset cardinality: {train.cardinality().numpy()}")
    print(f"Test dataset cardinality: {val.cardinality().numpy()}")
    print(f"Val dataset cardinality: {test.cardinality().numpy()}")
    
    write_tfrecord(train, 'datasets/train_negative.tfrecord')
    write_tfrecord(val, 'datasets/val_negative.tfrecord')
    write_tfrecord(test, 'datasets/test_negative.tfrecord')

    print("Datasets saved as TFRecord files in datasets folder")