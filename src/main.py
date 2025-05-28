from impl.Pipeline import Pipeline
from impl.NN import NeuralNetwork
from impl.Optimizers import ClassicOptimizer , AdaptiveLearningRateOptimizer , MomentumOptimizer , FracOptimizer , FracOptimizer2 , AdamOptimizer
from impl.CostFunctions import BinaryCrossEntropy , L2Regularization
from scipy.io import loadmat
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import json

def one_hot(y):
    one_hot = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        one_hot[i][y[i][0]-1] = 1
    return one_hot

def main():
    mat = loadmat("src/nn/ex3data1.mat")
    X = mat["X"]
    y = mat["y"]
    y = one_hot(y)
        
    p_gen = lambda Optimizer,params,output: Pipeline(
        X, 
        y, 
        NeuralNetwork(
            [25], 
            400, 
            10, 
            BinaryCrossEntropy(
                regularization=L2Regularization(0.2)
            ), 
            Optimizer(**params)
        ),
        output
    )
    
    D = [
        ( ClassicOptimizer, {"learning_rate":1}, "output/classical/"),
        ( AdaptiveLearningRateOptimizer, {"initial_learning_rate":1}, "output/adaptive/"),
        ( MomentumOptimizer, {"learning_rate":1, "momentum":0.5}, "output/momentum/"),
        ( FracOptimizer, {"learning_rate":1}, "output/frac/"),
        ( FracOptimizer2, {"learning_rate":1}, "output/frac2/"),
        ( AdamOptimizer, {"learning_rate":1}, "output/adam/")
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
    y_heigth = 100
    for Optimizer , _ , output in D:
        history = json.load(open(output + "history.json"))
        plt.plot(history["cost"], label=Optimizer.__name__)
        if history["cost"][10] < y_heigth:
            y_heigth = history["cost"][10]
    plt.ylim(ymin=0, ymax=y_heigth)
    plt.legend()
    plt.savefig("output/history.png")
    
if __name__ == "__main__":
    main()    
    