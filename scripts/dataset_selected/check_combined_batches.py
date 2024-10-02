from utils import load_combined_tfrecord

def print_batches(final_dataset):
    for batch_num, (volume, label) in enumerate(final_dataset):
        print(f"Batch {batch_num + 1}:")
        print(f"Volume shape: {volume.shape}")
        print(f"Labels: {label.numpy()}")

final_train_dataset = load_combined_tfrecord('data/final_train.tfrecord')
final_val_dataset = load_combined_tfrecord('data/final_val.tfrecord')
final_test_dataset = load_combined_tfrecord('data/final_test.tfrecord')

print("Train data:\n")
print_batches(final_train_dataset)
print("\nValidation data:\n")
print_batches(final_val_dataset)
print("\nTest data:\n")
print_batches(final_test_dataset)