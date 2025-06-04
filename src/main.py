from impl.Pipeline import Pipeline
from impl.NN import NeuralNetwork
from impl.Optimizers import ClassicOptimizer , AdaptiveLearningRateOptimizer , MomentumOptimizer , FracOptimizer , FracOptimizer2 , AdamOptimizer , FracAdap , Frac3Optimizer, FracTrue
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
        # ( MomentumOptimizer, {"learning_rate":1, "momentum":0.5}, "output/momentum/"),
        # ( FracOptimizer, {"learning_rate":1}, "output/frac/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.1}, "output/fracB01/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.01}, "output/fracB001/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.001}, "output/fracB0001/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":10}, "output/fracB10/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.5}, "output/fracB05/"),
        ( FracAdap, {"learning_rate":1,"beta":0.5}, "output/fracAdapB05/"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.5}, "output/frac3B05/"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.05}, "output/frac3B005/"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.005}, "output/frac3B0005/"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":5}, "output/frac3B5/"),
        # ( FracOptimizer2, {"learning_rate":1}, "output/frac2/"),
        ( FracOptimizer2, {"learning_rate":1,"beta":0}, "output/frac2B0/"),
        ( FracOptimizer2, {"learning_rate":1,"beta":0.1}, "output/frac2B01/"),
        ( FracOptimizer2, {"learning_rate":1,"beta":5}, "output/frac2B5/"),
        ( FracTrue, {"beta":0.5,"verbose":True}, "output/fracTrue/"),
        # ( AdamOptimizer, {"learning_rate":1}, "output/adam/")
    ]
    
    def run_pipeline(Optimizer,params,output):
        p = p_gen(Optimizer,params,output)
        p.run(epochs=1000,verbose=True)
    
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
    S = 20
    for Optimizer , _ , output in D:
        history = json.load(open(output + "history.json"))
        name = output.split("/")[-2]
        plt.plot(history["cost"], label=name)
        if history["cost"][S] < y_heigth:
            y_heigth = history["cost"][S]
    plt.ylim(ymin=0, ymax=y_heigth)
    plt.legend()
    plt.savefig("output/history.png")
    
    # similar plot but include x = time and y = cost
    plt.figure(figsize=(12, 8))
    plt.xlabel("Time")
    plt.ylabel("$J(\\Theta)$")
    plt.title("Cost function using Gradient Descent")
    plt.tight_layout()
    y_heigth = 0
    S = 20
    for Optimizer , _ , output in D:
        history = json.load(open(output + "history.json"))
        name = output.split("/")[-2]
        plt.plot(history["time"], history["cost"], label=name)
        if history["cost"][S] > y_heigth:
            y_heigth = history["cost"][S]
    plt.ylim(ymin=0, ymax=y_heigth)
    plt.legend()
    plt.savefig("output/history_time.png")
    
if __name__ == "__main__":
    main()    
    