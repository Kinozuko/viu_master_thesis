import sys
import os
sys.path.append('/home/kurose/Desktop/master/viu_master_thesis')

from scripts.dataset_selected.utils import parse_combined_tfrecord_fn, load_combined_tfrecord, count_records
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.dirname(current_dir) 

train_file_path = os.path.join(base_path, 'data', 'final_train.tfrecord')
val_file_path = os.path.join(base_path, 'data', 'final_val.tfrecord')
test_file_path = os.path.join(base_path, 'data', 'final_test.tfrecord')

def load_dataset(batch_size = 2):
    # Load train, validation, and test datasets

    train_dataset = load_combined_tfrecord(train_file_path,batch_size=batch_size, add_channel=True,shuffle=True,buffer_size=20)
    val_dataset = load_combined_tfrecord(val_file_path, batch_size=batch_size, add_channel=True,shuffle=True,buffer_size=20)
    test_dataset = load_combined_tfrecord(test_file_path, batch_size=batch_size, add_channel=True,shuffle=True,buffer_size=20)
    return train_dataset, val_dataset, test_dataset

def get_input_shape(train): 
    for i in train:
        input_shape = i[0].shape[1:]
        break
    return input_shape

def f1_score(y_true, y_pred):
    # Cast y_true to float32 to match y_pred type
    y_true = tf.cast(y_true, tf.float32)
    
    y_pred = tf.round(y_pred)
    
    # Calculate True Positives, False Positives, and False Negatives
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float32'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float32'), axis=0)
    
    # Precision and Recall
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    # Taking the mean over the batch
    f1 = tf.reduce_mean(f1)  
    
    return f1

def save_and_print_model(model, file_path="model_image.png"):
    model.summary()
    plot_model(model, to_file=file_path, show_shapes=True, show_layer_names=True)
    print(f"Model architecture saved to {file_path}")

def get_callbacks(
    checkpoint_path='best_model.h5',
    monitor_metric='val_loss',
    patience=10,
    reduce_factor=0.1,
    reduce_patience=5,
    reduce_min_lr=1e-6):

    # Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        restore_best_weights=True
    )

    # Model Checkpoint
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor_metric,
        save_best_only=True,
        save_weights_only=True
    )

    # Reduce LR On Plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=reduce_factor,
        patience=reduce_patience,
        min_lr=reduce_min_lr
    )

    return [early_stopping, model_checkpoint, reduce_lr]

def get_metrics():
    return ['accuracy', tf.keras.metrics.AUC(name='auc'), f1_score]

def plot_metrics(history, save_path=None):
    epochs = range(1, len(history.history['loss']) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left plot: Accuracy and Validation Accuracy
    axs[0, 0].plot(epochs, history.history['accuracy'], 'bo-', label='Training Accuracy')
    axs[0, 0].plot(epochs, history.history['val_accuracy'], 'ro-', label='Validation Accuracy')
    axs[0, 0].set_title('Training and Validation Accuracy')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].legend()

    # Top-right plot: Loss and Validation Loss
    axs[0, 1].plot(epochs, history.history['loss'], 'bo-', label='Training Loss')
    axs[0, 1].plot(epochs, history.history['val_loss'], 'ro-', label='Validation Loss')
    axs[0, 1].set_title('Training and Validation Loss')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()

    # Bottom-left plot: F1 Score and Validation F1 Score
    axs[1, 0].plot(epochs, history.history['f1_score'], 'bo-', label='Training F1 Score')
    axs[1, 0].plot(epochs, history.history['val_f1_score'], 'ro-', label='Validation F1 Score')
    axs[1, 0].set_title('Training and Validation F1 Score')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('F1 Score')
    axs[1, 0].legend()

    # Bottom-right plot: AUC and Validation AUC
    axs[1, 1].plot(epochs, history.history['auc'], 'bo-', label='Training AUC')
    axs[1, 1].plot(epochs, history.history['val_auc'], 'ro-', label='Validation AUC')
    axs[1, 1].set_title('Training and Validation AUC')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('AUC')
    axs[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()