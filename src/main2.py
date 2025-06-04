from impl.Pipeline import Pipeline
from impl.NN import NeuralNetwork
from impl.Optimizers import ClassicOptimizer , AdaptiveLearningRateOptimizer , MomentumOptimizer , FracOptimizer , FracOptimizer2 , AdamOptimizer , FracAdap , Frac3Optimizer
from impl.CostFunctions import BinaryCrossEntropy , L2Regularization
from scipy.io import loadmat
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import json
import glob
import os
from pathlib import Path

BASE_DIR = "output2/"
DATASET_PATH = "src/nn/1/"

def one_hot(y):
    one_hot = np.zeros((y.shape[0], y.max() + 1))
    for i in range(y.shape[0]):
        one_hot[i][y[i][0]] = 1
    return one_hot

def load_dataset():
    # labels is the directory names in DATASET_PATH
    labels = [d.name for d in os.scandir(DATASET_PATH) if d.is_dir()]
    labels_map = {label: i for i, label in enumerate(labels)}
    images = []
    labels_list = []
    # open each directory in DATASET_PATH and load the images png
    for label in labels:
        label_path = os.path.join(DATASET_PATH, label)
        for img_path in glob.glob(os.path.join(label_path, "*.png")):
            img = plt.imread(img_path)
            images.append(img.flatten())
            labels_list.append(int(labels_map[label]))
    return np.array(images), np.array(labels_list) , labels_map
    
def main():
    # Load dataset
    X, y , labels = load_dataset()
    print(X.shape, y.shape, labels)
    y = one_hot(y.reshape(-1, 1))
        
    p_gen = lambda Optimizer,params,output: Pipeline(
        X, 
        y, 
        NeuralNetwork(
            [25], 
            X.shape[1], 
            7, 
            BinaryCrossEntropy(
                regularization=L2Regularization(0.2)
            ), 
            Optimizer(**params)
        ),
        output
    )
    
    D = [
        ( ClassicOptimizer, {"learning_rate":1}, BASE_DIR + "classical/"),
        ( AdaptiveLearningRateOptimizer, {"initial_learning_rate":1}, BASE_DIR + "adaptive/"),
        # ( MomentumOptimizer, {"learning_rate":1, "momentum":0.5}, BASE_DIR + "momentum/"),
        ( FracOptimizer, {"learning_rate":1}, BASE_DIR + "frac/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.1}, BASE_DIR + "fracB01/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.01}, BASE_DIR + "fracB001/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.001}, BASE_DIR + "fracB0001/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":10}, BASE_DIR + "fracB10/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.5}, BASE_DIR + "fracB05/"),
        ( FracAdap, {"learning_rate":1,"beta":1}, BASE_DIR + "fracAdapB1/"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.5}, BASE_DIR + "frac3B05/"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.05}, BASE_DIR + "frac3B005/"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.005}, BASE_DIR + "frac3B0005/"),
        ( Frac3Optimizer, {"learning_rate":1,"beta":5}, BASE_DIR + "frac3B5/"),
        # ( FracOptimizer2, {"learning_rate":1}, BASE_DIR + "frac2/"),
        # ( FracOptimizer2, {"learning_rate":1,"beta":0}, BASE_DIR + "frac2B0/"),
        # ( FracOptimizer2, {"learning_rate":1,"beta":0.1}, BASE_DIR + "frac2B01/"),
        # ( FracOptimizer2, {"learning_rate":1,"beta":5}, BASE_DIR + "frac2B5/"),
        # ( AdamOptimizer, {"learning_rate":1}, BASE_DIR + "adam/")
    ]
    
    def run_pipeline(Optimizer,params,output):
        p = p_gen(Optimizer,params,output)
        p.run(epochs=1000)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_pipeline, Optimizer,params,output) for Optimizer,params,output in D]
        for future in futures:
            future.result()
    
    # open all history files and plot them
    plt.figure(figsize=(12, 8))
    plt.xlabel("Iteration")
    plt.ylabel("$J(\\Theta)$")
    plt.title("Cost function using Gradient Descent")
    plt.tight_layout()
    y_heigth = 0
    S = 20
    for Optimizer , _ , output in D:
        history = json.load(open(output + "history.json"))
        name = output.split("/")[-2]
        plt.plot(history["cost"], label=name)
        if history["cost"][S] > y_heigth:
            y_heigth = history["cost"][S]
    plt.ylim(ymin=0.5, ymax=y_heigth)
    plt.legend()
    plt.savefig(BASE_DIR + "history.png")
    
if __name__ == "__main__":
    main()    
    