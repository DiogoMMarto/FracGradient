from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets , layers , models
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
# tf.config.run_functions_eagerly(True)

from ciphar10.Pipeline import Pipeline, end_graphs
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

DATASET_PATH = "datasets/ex3data1.mat"
BASE_DIR = "results/output_MNIST_4/"
BATCH_SIZE = 10
NUM_CLASSES = 10
DATA_AUGMENTATION = False
os.makedirs(BASE_DIR, exist_ok=True)
NUM_EPOCHS = 5
VERBOSE = False
EXPERIMENT_NAME = "MNIST 4 - CNN"

def one_hot(y):
    one_hot = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        one_hot[i][y[i][0]-1] = 1
    return one_hot

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

@tf.function
def alpha_function5(norm_GradCost, beta):
    """
    Computes the alpha value based on the norm of the gradient cost.
    """
    return 2 - 1.0 / ( 1.0 + (norm_GradCost * beta))

def load_dataset():
    # use tensorflow datasets to load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    labels = np.arange(NUM_CLASSES)
    print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")
    return X_train, y_train, X_test, y_test, labels

def create_model():
    activation = 'tanh'
    m =  models.Sequential([
        # input 28x28x1
        layers.InputLayer(input_shape=(28, 28, 1)),
        layers.Conv2D(6, (5, 5), strides=1, activation=activation, padding="same"),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Conv2D(16, (5, 5), strides=1, activation=activation, padding="valid"),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Conv2D(120, (5, 5), strides=1, activation=activation, padding="valid"),
        layers.Flatten(),
        layers.Dense(84, activation=activation),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    m.summary()
    # initialize the weights to be from -0.1 to 0.1
    for layer in m.layers:
        if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):
            weights = layer.get_weights()
            if len(weights) > 0:
                weights[0] = np.random.uniform(-0.1, 0.1, size=weights[0].shape)
                if len(weights) > 1:
                    weights[1] = np.random.uniform(-0.1, 0.1, size=weights[1].shape)
                layer.set_weights(weights)
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
        dataset_name="MNIST"  ,
        expirement_name=EXPERIMENT_NAME,
    )
    
    D = [
        # (FracOptimizer(learning_rate=0.03,beta=0.5), "FracOptimizer B=0.5"),
        # (FracOptimizer(learning_rate=0.03,beta=0.05), "FracOptimizer B=0.05"),
        (FracOptimizer(learning_rate=0.1,beta=0.005), "FracOptimizer B=0.005"),
        # (FracOptimizer(learning_rate=0.03,beta=0.0005,alpha_func=alpha_function5), "FracOptimizer B=0.0005 alpha5"),
        # (FracOptimizer(learning_rate=0.03,beta=0.005,alpha_func=alpha_function5), "FracOptimizer B=0.005 alpha5"),
        # (FracOptimizer(learning_rate=0.03,beta=0.05,alpha_func=alpha_function5), "FracOptimizer B=0.05 alpha5"),
        # (FracOptimizer(learning_rate=0.03,beta=0.01,alpha_func=alpha_function2), "FracOptimizer B=0.01 alpha2"),
        # (FracOptimizer(learning_rate=0.03,beta=0.05,alpha_func=alpha_function3), "FracOptimizer B=0.05 alpha3"),          
        # (FracOptimizer(learning_rate=0.03,beta=0.005,alpha_func=alpha_function4), "FracOptimizer B=0.005 alpha4"),          
        # (FracOptimizer(learning_rate=0.03,beta=0.01), "FracOptimizer B=0.01"),
        # (tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0001), "SGD 0.001"),
        # (tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0001), "SGD"),
        # (tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0001), "SGD 0.1"),
        # (tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.0001), "SGD 1"),
        # (tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.0001), "SGD 0.05"),
        # (tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.0001), "SGD"),
        (tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0001), "SGD"),
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
    
    end_graphs(BASE_DIR,D,EXPERIMENT_NAME)
    
if __name__ == "__main__":
    main()    
    