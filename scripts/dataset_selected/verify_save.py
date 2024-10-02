import matplotlib.pyplot as plt
from utils import load_tfrecord

# Function to visualize the middle slice from a 3D volume
def get_middle_slice(volume):
    middle_index = volume.shape[0] // 2  # Get the middle slice index
    return volume[middle_index]

# Function to visualize one volume from a dataset
def visualize_middle_slice_from_splits(train_dataset, val_dataset, test_dataset):
    # Initialize the plot for 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Extract from Train
    for batch in train_dataset.take(1):
        train_volume = batch[0].numpy() 
        middle_slice_train = get_middle_slice(train_volume)
        axs[0].imshow(middle_slice_train, cmap='gray')
        axs[0].set_title("Train - Middle Slice")
        axs[0].axis('off')
        break

    # Extract from Val
    for batch in val_dataset.take(1):
        val_volume = batch[0].numpy()
        middle_slice_val = get_middle_slice(val_volume)
        axs[1].imshow(middle_slice_val, cmap='gray')
        axs[1].set_title("Validation - Middle Slice")
        axs[1].axis('off')
        break

    # Extract from Test
    for batch in test_dataset.take(1):
        test_volume = batch[0].numpy()
        middle_slice_test = get_middle_slice(test_volume)
        axs[2].imshow(middle_slice_test, cmap='gray')
        axs[2].set_title("Test - Middle Slice")
        axs[2].axis('off')
        break

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig("datasets/load_show.png")
    plt.show()

if __name__ == '__main__':
    train = load_tfrecord('datasets/train.tfrecord').batch(2)
    val = load_tfrecord('datasets/val.tfrecord').batch(2)
    test = load_tfrecord('datasets/test.tfrecord').batch(2)

    visualize_middle_slice_from_splits(train, val, test)