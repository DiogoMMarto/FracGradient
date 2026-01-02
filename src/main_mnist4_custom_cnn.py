"""
MNIST 4 – Custom CNN (Keras/TensorFlow)

Trains a LeNet-style CNN on the MNIST dataset and compares different optimizers,
including the fractional FracOptimizer. Each optimizer is run through a shared
Pipeline, and results/plots are saved under `results/output_MNIST_4_custom_cnn/`.

What this script does
- Loads MNIST from `tf.keras.datasets` and normalizes pixel values.
- Builds a CNN architecture similar to LeNet-5 with tanh activations:
  Conv → Pool → Conv → Pool → Conv → Dense(84) → Softmax(10).
- Initializes Conv/Dense weights uniformly in [-0.1, 0.1] for reproducibility.
- Runs training with the optimizers listed in `D` and aggregates results via `end_graphs`.

How to customize
- Choose optimizers: comment/uncomment entries in the `D` list to include/exclude them.
  (Examples include FracOptimizer with various β, SGD at different learning rates, etc.)
- Experiment with α(N): pass one of the alpha functions into FracOptimizer via `alpha_func=...`.
- Modify training: adjust `NUM_EPOCHS`, `BATCH_SIZE`, learning rates, and β values.
- Output: change `BASE_DIR` and `EXPERIMENT_NAME` to save results in a different folder.

Notes
- Uses batch size = 10 and trains for 5 epochs by default (as in the original experiment setup).
- Labels are one-hot encoded with 10 classes.
"""
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets , layers , models
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
# tf.config.run_functions_eagerly(True)

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
BASE_DIR = "results/output_MNIST_4_custom_cnn/"
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
        output_dir=BASE_DIR + Name_Optimizer.replace(" ","_").replace("$\\beta$","B") + "/",
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
        # (FracOptimizer(learning_rate=0.03,beta=0.5), "FracGradient V2 $\\beta$=0.5"),
        # (FracOptimizer(learning_rate=0.03,beta=0.05), "FracGradient V2 $\\beta$=0.05"),
        (FracOptimizer(learning_rate=0.1,beta=0.005), "FracGradient V2 $\\beta$=0.005"),
        # (FracOptimizer(learning_rate=0.03,beta=0.0005,alpha_func=alpha_function5), "FracGradient V2 $\\beta$=0.0005 alpha5"),
        # (FracOptimizer(learning_rate=0.03,beta=0.005,alpha_func=alpha_function5), "FracGradient V2 $\\beta$=0.005 alpha5"),
        # (FracOptimizer(learning_rate=0.03,beta=0.05,alpha_func=alpha_function5), "FracGradient V2 $\\beta$=0.05 alpha5"),
        # (FracOptimizer(learning_rate=0.03,beta=0.01,alpha_func=alpha_function2), "FracGradient V2 $\\beta$=0.01 alpha2"),
        # (FracOptimizer(learning_rate=0.03,beta=0.05,alpha_func=alpha_function3), "FracGradient V2 $\\beta$=0.05 alpha3"),          
        # (FracOptimizer(learning_rate=0.03,beta=0.005,alpha_func=alpha_function4), "FracGradient V2 $\\beta$=0.005 alpha4"),          
        # (FracOptimizer(learning_rate=0.03,beta=0.01), "FracGradient V2 $\\beta$=0.01"),
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
    