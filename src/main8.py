from tensorflow.keras import datasets , layers , models
import tensorflow as tf
import numpy as np
import h5py
# tf.config.run_functions_eagerly(True)

from ciphar10.Pipeline import Pipeline
from ciphar10.Optimizer import FracOptimizer

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
else:
    print("No GPU found, using CPU.")

import matplotlib.pyplot as plt
import json
from pathlib import Path
import os

#   0: gamma -> index 0
#   1: beta -> index 1
#   2: kernel -> index 2
#   3: bias -> index 3
#   4: kernel -> index 4
#   5: bias -> index 5
#   6: kernel -> index 6
#   7: bias -> index 7
#   8: kernel -> index 8
#   9: bias -> index 9

DATASET_PATH = "datasets/Happy_datasets/datasets/"
BATCH_SIZE = 64 * 2
NUM_CLASSES = 2
DATA_AUGMENTATION = False
BASE_DIR = "results/output_HappyFace_5/"
os.makedirs(BASE_DIR, exist_ok=True)
NUM_EPOCHS = 100
VERBOSE = True

@tf.function
def alpha_function2(norm_GradCost, beta):
    """
    Computes the alpha value based on the norm of the gradient cost.
    """
    return 1.0 / ( 1.0 + (norm_GradCost * beta))

@tf.function
def alpha_function3(norm_GradCost, beta):
    """
    Computes the alpha value based on the norm of the gradient cost.
    """
    return 1.0 / tf.cos(norm_GradCost * beta)

@tf.function
def alpha_function4(norm_GradCost, beta):
    """
    Computes the alpha value based on the norm of the gradient cost.
    """
    return 2.0 / ( 1.0 + tf.exp(norm_GradCost * beta))

def one_hot(y):
    one_hot = np.zeros((y.shape[0], y.max() + 1))
    for i in range(y.shape[0]):
        one_hot[i][y[i]] = 1
    return one_hot
    
def load_dataset():
    # .h5 format on test_happy.h5 and train_happy.h5
    train_path = Path(DATASET_PATH) / "train_happy.h5"
    test_path = Path(DATASET_PATH) / "test_happy.h5"
    with h5py.File(train_path, 'r') as f:
        X_train: ndarray = f['train_set_x'][:] # type: ignore
        y_train: ndarray = f['train_set_y'][:]  # type: ignore
        list_classes: list = f['list_classes'][:]  # type: ignore
        
    with h5py.File(test_path, 'r') as f:
        X_test: ndarray = f['test_set_x'][:]  # type: ignore
        y_test: ndarray = f['test_set_y'][:]  # type: ignore
        list_classes_test: list = f['list_classes'][:]  # type: ignore
        
    # convert into format that tensorflow can use in cnn
    
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)
    # check if the classes are the same
    if not np.array_equal(list_classes, list_classes_test):
        raise ValueError("The classes in the train and test sets are not the same.")

    return X_train , y_train, X_test, y_test, list_classes

def create_model():
    m =  models.Sequential([
        layers.BatchNormalization(input_shape=(64, 64, 3)),
        
        layers.Conv2D(32, (4, 4), padding='same'),
        layers.MaxPooling2D((4, 4)),
        layers.Dropout(0.1),
        
        layers.Conv2D(64, (4, 4), padding='same'),
        layers.MaxPooling2D((4, 4)),
        layers.Dropout(0.1),

        # layers.Conv2D(128, (3, 3), padding='same'),
        # layers.MaxPooling2D((2, 2)),
        # layers.Dropout(0.1),
        
        # layers.Conv2D(256, (3, 3), padding='same'),
        # layers.MaxPooling2D((2, 2)),
        # layers.Dropout(0.1),

        layers.Flatten(),
        layers.Dense(256),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    m.summary()
    return m

def main():
    # Load dataset
    X, y, X_test, y_test, labels = load_dataset()

    p_gen = lambda Optimizer, Name_Optimizer : Pipeline(
        X,
        y,
        model=create_model(),
        name= Name_Optimizer,
        compile_kwargs={
            "loss": tf.keras.losses.CategoricalCrossentropy(),
            "optimizer": Optimizer,
            "metrics": ["accuracy"]
        },
        output_dir=BASE_DIR + Name_Optimizer.replace(" ","_") + "/",
        X_test=X_test,
        y_test=y_test,
        data_augmentation=DATA_AUGMENTATION,
        overwrite=False,
        continue_training= False ,
        batch_size=BATCH_SIZE,
        dataset_name="HappyFace" 
    )
    
    D = [
        # (FracOptimizer(learning_rate=0.03,beta=0.5), "FracOptimizer B=0.5"),
        (FracOptimizer(learning_rate=0.03,beta=0.05), "FracOptimizer B=0.05"),
        (FracOptimizer(learning_rate=0.03,beta=0.005), "FracOptimizer B=0.005"),
        (FracOptimizer(learning_rate=0.03,beta=0.005,alpha_func=alpha_function2), "FracOptimizer B=0.005 alpha2"),
        # (FracOptimizer(learning_rate=0.03,beta=0.01,alpha_func=alpha_function2), "FracOptimizer B=0.01 alpha2"),
        # (FracOptimizer(learning_rate=0.03,beta=0.05,alpha_func=alpha_function3), "FracOptimizer B=0.05 alpha3"),          
        (FracOptimizer(learning_rate=0.03,beta=0.005,alpha_func=alpha_function4), "FracOptimizer B=0.005 alpha4"),          
        # (FracOptimizer(learning_rate=0.03,beta=0.01), "FracOptimizer B=0.01"),
        # (tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0001), "SGD 0.001"),
        # (tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0001), "SGD"),
        # (tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0001), "SGD 0.1"),
        # (tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.0001), "SGD 1"),
        # (tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.0001), "SGD 0.05"),
        (tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.0001), "SGD"),
        # (tf.keras.optimizers.SGD(learning_rate=0.5, momentum=0.0001), "SGD 0.5"),
        (tf.keras.optimizers.Adam(), "Adam"),
        (tf.keras.optimizers.RMSprop(), "RMSprop"),
    ]
    
    def run_pipeline(Optimizer,Name_Optimizer):
        p = p_gen(Optimizer, Name_Optimizer)
        p.run(epochs=NUM_EPOCHS,verbose=VERBOSE)
    
    for Optimizer, Name_Optimizer in D:
        run_pipeline(Optimizer, Name_Optimizer)
    
    print("All pipelines completed.")
    
    # open all model directories
    # load the cost function history from each model
    # plot the cost function history for each model in the same plot
    plt.figure(figsize=(12, 8))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss function history")
    plt.tight_layout()
    for Optimizer, Name_Optimizer in D:
        output_dir = BASE_DIR + Name_Optimizer.replace(" ","_") + "/"
        history_path = Path(output_dir) / "history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            plt.plot(history['loss'], label=Name_Optimizer)
        else:
            print(f"History file not found for {Name_Optimizer} at {history_path}")
    plt.legend()
    plt.savefig(BASE_DIR + "loss_history.png")
    print(f"Loss history saved to {BASE_DIR}loss_history.png")
    
    # similar plot but include x = time and y = cost
    plt.figure(figsize=(12, 8))
    plt.xlabel("Time (seconds)")
    plt.ylabel("Loss")
    plt.title("Loss function history over time")
    plt.tight_layout()
    for Optimizer, Name_Optimizer in D:
        output_dir = BASE_DIR + Name_Optimizer.replace(" ","_") + "/"
        history_path = Path(output_dir) / "history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            cumulative_time = [sum(history['time'][:i+1]) for i in range(len(history['time']))]
            plt.plot(cumulative_time, history['loss'], label=Name_Optimizer)
        else:
            print(f"History file not found for {Name_Optimizer} at {history_path}")
    plt.legend()
    plt.savefig(BASE_DIR + "loss_history_time.png")
    print(f"Loss history over time saved to {BASE_DIR}loss_history_time.png")
    
    # for cost function history, plot the cost function history for each model in the same plot
    plt.figure(figsize=(12, 8))
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Cost function history")
    plt.tight_layout()
    for Optimizer, Name_Optimizer in D:
        output_dir = BASE_DIR + Name_Optimizer.replace(" ","_") + "/"
        history_path = Path(output_dir) / "history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            plt.plot(history['loss'], label=Name_Optimizer)
        else:
            print(f"History file not found for {Name_Optimizer} at {history_path}")
    plt.legend()
    plt.savefig(BASE_DIR + "cost_history.png")
    print(f"Cost history saved to {BASE_DIR}cost_history.png")
    
    # for validation cost function history, plot the validation cost function history for each model in the same plot
    plt.figure(figsize=(12, 8))
    plt.xlabel("Epoch")
    plt.ylabel("Validation Cost")
    plt.title("Validation Cost function history")
    plt.tight_layout()
    for Optimizer, Name_Optimizer in D:
        output_dir = BASE_DIR + Name_Optimizer.replace(" ","_") + "/"
        history_path = Path(output_dir) / "history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            plt.plot(history['val_loss'], label=Name_Optimizer)
        else:
            print(f"History file not found for {Name_Optimizer} at {history_path}")
    plt.legend()
    plt.savefig(BASE_DIR + "val_cost_history.png")
    print(f"Validation cost history saved to {BASE_DIR}val_cost_history.png")
    
    plt.figure(figsize=(12, 8))
    plt.xlabel("Epoch")
    plt.ylabel("ValidationAccuracy")
    plt.title("Validation Accuracy history")
    plt.tight_layout()
    for Optimizer, Name_Optimizer in D:
        output_dir = BASE_DIR + Name_Optimizer.replace(" ","_") + "/"
        history_path = Path(output_dir) / "history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            plt.plot(history['val_accuracy'], label=Name_Optimizer)
        else:
            print(f"History file not found for {Name_Optimizer} at {history_path}")
    plt.legend()
    plt.savefig(BASE_DIR + "val_accuracy_history.png")
    print(f"Validation accuracy history saved to {BASE_DIR}val_accuracy_history.png")
    
    plt.figure(figsize=(12, 8))
    plt.xlabel("Time (seconds)")
    plt.ylabel("Loss")
    plt.title("Loss function history over time")
    plt.tight_layout()
    for Optimizer, Name_Optimizer in D:
        output_dir = BASE_DIR + Name_Optimizer.replace(" ","_") + "/"
        history_path = Path(output_dir) / "history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            cumulative_time = [sum(history['time'][:i+1]) for i in range(len(history['time']))]
            plt.plot(cumulative_time, history['val_loss'], label=Name_Optimizer)
        else:
            print(f"History file not found for {Name_Optimizer} at {history_path}")
    plt.legend()
    plt.savefig(BASE_DIR + "val_loss_history_time.png")
    print(f"Loss history over time saved to {BASE_DIR}val_loss_history_time.png")
    
if __name__ == "__main__":
    main()    
    