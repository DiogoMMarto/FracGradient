"""
This script trains a custom CNN on the CIFAR-10 dataset (data augmentation) 
using different optimizers (FracOptimizer, SGD, Adam, RMSprop). 
It runs each optimizer through a common training pipeline and saves results/plots.  

To customize: 
- Comment/uncomment optimizers in the `D` list to choose which ones to include in the experiment.
- Adjust constants (e.g., NUM_EPOCHS, BATCH_SIZE, learning_rate, beta) to modify training behavior.
- Results are saved under `results/output_cifar10_custom_cnn_data_augmentation/`.
"""
from tensorflow.keras import datasets , layers , models
import tensorflow as tf

from ciphar10.Pipeline import Pipeline, end_graphs
from ciphar10.optimizer import FracOptimizer

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
else:
    print("No GPU found, using CPU.")

import os

BATCH_SIZE = 256
NUM_CLASSES = 10
DATA_AUGMENTATION = True
BASE_DIR = "results/output_cifar10_custom_cnn_data_augmentation_2/"
os.makedirs(BASE_DIR, exist_ok=True)
NUM_EPOCHS = 150
VERBOSE = True
EXPIREMENT_NAME = "CIFAR-10 CNN with Data Augmentation"

def load_dataset():
    (X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    return X_train, y_train, X_test, y_test, datasets.cifar10.load_data()[0][1].tolist()

def create_model():
    return models.Sequential([
        layers.BatchNormalization(input_shape=(32, 32, 3)),
        
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),
        
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),
        
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Flatten(),
        layers.Dense(512),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

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
        output_dir=BASE_DIR + Name_Optimizer.replace(" ","_").replace("$\\beta$","B") + "/",
        X_test=X_test,
        y_test=y_test,
        data_augmentation=DATA_AUGMENTATION,
        overwrite=False,
        continue_training= False ,
        batch_size=BATCH_SIZE,
        dataset_name="CIFAR-10" ,
        expirement_name=EXPIREMENT_NAME,
    )
    
    D = [
        # (FracOptimizer(learning_rate=0.01,beta=0.05), "FracGradient V2 $\\beta$=0.05"),     
        (tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0001), "SGD"),
        (tf.keras.optimizers.Adam(), "Adam"),
        (tf.keras.optimizers.RMSprop(), "RMSprop"),
        (FracOptimizer(learning_rate=0.01,beta=0.005), "FracGradient V2 $\\beta$=0.005"),
        (FracOptimizer(learning_rate=0.01,beta=0.01), "FracGradient V2 $\\beta$=0.01"),
    ]
    
    def run_pipeline(Optimizer,Name_Optimizer):
        p = p_gen(Optimizer, Name_Optimizer)
        p.run(epochs=NUM_EPOCHS,verbose=VERBOSE)
    
    for Optimizer, Name_Optimizer in D:
        run_pipeline(Optimizer, Name_Optimizer)
    
    print("All pipelines completed.")
    
    end_graphs(BASE_DIR,D,EXPIREMENT_NAME)
    
if __name__ == "__main__":
    main()    
    