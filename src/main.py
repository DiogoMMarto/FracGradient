from impl.Pipeline import Pipeline, gen_grid_search
from impl.NN import NeuralNetwork
from impl.Optimizers import ClassicOptimizer , AdaptiveLearningRateOptimizer , MomentumOptimizer , FracOptimizer , FracOptimizer2 , AdamOptimizer , FracAdap , Frac3Optimizer, FracTrue , FracOptimizerBStable
from impl.CostFunctions import BinaryCrossEntropy , L2Regularization , ActivationFunction
from scipy.io import loadmat
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split

DATASET_PATH = "datasets/ex3data1.mat"
BASE_DIR = "results/output_MNIST/"
NUM_EPOCHS = 1000
VERBOSE = True

def one_hot(y):
    one_hot = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        one_hot[i][y[i][0]-1] = 1
    return one_hot

def main():
    mat = loadmat(DATASET_PATH)
    X = mat["X"]
    y = mat["y"]
    y = one_hot(y)
    
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    p_gen = lambda Optimizer,params,output: Pipeline(
        X, 
        y, 
        NeuralNetwork(
            [25], 
            400, 
            10, 
            BinaryCrossEntropy(
                regularization=L2Regularization(0.2),
                activation_function_names=[
                    "sigmoid",
                    "sigmoid",
                ]
            ), 
            Optimizer(**params)
        ),
        output,
        X_test=X_test,
        y_test=y_test
    )
    
    D = [
        ( ClassicOptimizer, {"learning_rate":1}, BASE_DIR + "classical/" , "Gradient Descent"),
        ( AdaptiveLearningRateOptimizer, {"initial_learning_rate":1}, BASE_DIR + "adaptive/" , "Adaptive Learning Rate"),
        # ( MomentumOptimizer, {"learning_rate":1, "momentum":0.5}, BASE_DIR + "momentum/"),
        # ( FracOptimizer, {"learning_rate":1}, BASE_DIR + "frac/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.1}, BASE_DIR + "fracB01/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.01}, BASE_DIR + "fracB001/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.001}, BASE_DIR + "fracB0001/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":10}, BASE_DIR + "fracB10/"),
        ( FracOptimizer, {"learning_rate":1,"beta":0.5}, BASE_DIR + "fracB05/" , "FracGradient V2"),
        ( FracAdap, {"learning_rate":1,"beta":0.5}, BASE_DIR + "fracAdapB05/", "FracGradient V2 Adaptive"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.5}, BASE_DIR + "frac3B05/" , "Fractional Gradient Descent V3"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.05}, BASE_DIR + "frac3B005/"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.005}, BASE_DIR + "frac3B0005/"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":5}, BASE_DIR + "frac3B5/"),
        # ( FracOptimizer2, {"learning_rate":1}, BASE_DIR + "frac2/"),
        # ( FracOptimizer2, {"learning_rate":1,"beta":0}, BASE_DIR + "frac2B0/"),
        ( FracOptimizer2, {"learning_rate":1,"beta":0.1}, BASE_DIR + "frac2B01/", "FracGradient"),
        # ( FracOptimizer2, {"learning_rate":1,"beta":5}, BASE_DIR + "frac2B5/"),
        # ( FracTrue, {"beta":0.5,"verbose":True}, BASE_DIR + "fracTrue/"),
        # ( AdamOptimizer, {"learning_rate":1}, BASE_DIR + "adam/"),
        # ( FracOptimizerBStable, {"learning_rate":1,"beta":0.5}, BASE_DIR + "fracBStable05/", "FracGradient B Stable"),
        ( FracOptimizerBStable, {"learning_rate":1,"beta":0.01}, BASE_DIR + "fracBStable001/", "FracGradient B Stable"),
        # ( FracOptimizerBStable, {"learning_rate":1,"beta":0.01}, BASE_DIR + "fracBStable001/", "FracGradient B Stable 0.01"),
        # ( FracOptimizerBStable, {"learning_rate":1,"beta":0.005}, BASE_DIR + "fracBStable0005/", "FracGradient B Stable 0.005"),
        # ( FracOptimizerBStable, {"learning_rate":1,"beta":5}, BASE_DIR + "fracBStable5/", "FracGradient B Stable 5"),
    ]
    
    D = gen_grid_search(
        [(FracOptimizer , {"learning_rate":[10,5,2,1,0.1,0.01,0.001],"beta":[5,1,0.5,0.1,0.05,0.01,0.005,0.001]}, BASE_DIR + "_frac_v2_/", "FracGradient V2"),
         (FracAdap , {"learning_rate":[5,2,1],"beta":[5,1,0.5,0.1,0.05,0.01]}, BASE_DIR + "_frac_adap_v2/", "FracGradient V2 Adaptive"),
         (FracOptimizer , {"learning_rate":[1],"beta":list(2**np.arange(-10,3,0.3))}, BASE_DIR + "_frac_v2_/", "FracGradient V2"),
         (FracAdap , {"learning_rate":[1],"beta":list(2**np.arange(-10,3,0.3))}, BASE_DIR + "_frac_adap_v2/", "FracGradient V2 Adaptive"),]
    )
    
    def run_pipeline(Optimizer,params,output):
        p = p_gen(Optimizer,params,output)
        p.run(epochs=NUM_EPOCHS,verbose=VERBOSE)
    
    if False:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_pipeline, Optimizer,params,output) for Optimizer,params,output,_ in D]
            for future in futures:
                future.result()
    else:
        for Optimizer, params, output, name in D:
            p = p_gen(Optimizer, params, output)
            p.run(epochs=NUM_EPOCHS, verbose=VERBOSE)
    
    # open all history files and plot them
    plt.figure(figsize=(12, 8))
    plt.xlabel("Iteration")
    plt.ylabel("$J(\\Theta)$")
    plt.title("Cost function using Gradient Descent")
    plt.tight_layout()
    y_heigth = 100
    S = 10
    for Optimizer , _ , output,name in D:
        history = json.load(open(output + "history.json"))
        # name = output.split("/")[-2]
        plt.plot(history["cost"], label=name)
        if history["cost"][S] < y_heigth:
            y_heigth = history["cost"][S]
    plt.ylim(ymin=0, ymax=y_heigth)
    plt.legend()
    plt.savefig(BASE_DIR + "history.png")
    
    # similar plot but include x = time and y = cost
    plt.figure(figsize=(12, 8))
    plt.xlabel("Time")
    plt.ylabel("$J(\\Theta)$")
    plt.title("Cost function using Gradient Descent")
    plt.tight_layout()
    y_heigth = 0
    S = 10
    for Optimizer , _ , output,name in D:
        history = json.load(open(output + "history.json"))
        # name = output.split("/")[-2]
        plt.plot(history["time"], history["cost"], label=name)
        if history["cost"][S] > y_heigth:
            y_heigth = history["cost"][S]
    plt.ylim(ymin=0, ymax=y_heigth)
    plt.legend()
    plt.savefig(BASE_DIR + "history_time.png")
    
    min_cost = float('inf')
    best_optimizer = None
    for Optimizer, _, output, name in D:
        history = json.load(open(output + "history.json"))
        final_cost = history["cost"][-1]
        if final_cost < min_cost:
            min_cost = final_cost
            best_optimizer = name
    print(f"The best optimizer is {best_optimizer} with a final cost of {min_cost:.4f}")
    
    # if the optimizers have params beta, plot the final cost vs beta
    plt.figure(figsize=(12, 8))
    plt.xlabel("Beta")
    # logscale x-axis
    plt.xscale("log")
    plt.ylabel("$J(\\Theta)$")
    # limit y-axis to [0, 1]
    plt.ylim(0.2, 0.5)
    plt.title("Final Cost vs Beta")
    plt.tight_layout()
    betas = []
    costs = []
    for Optimizer, params, output, name in D:
        if "beta" in params:
            history = json.load(open(output + "history.json"))
            final_cost = history["cost"][-1]
            betas.append(params["beta"])
            costs.append(final_cost)
    plt.scatter(betas, costs)
    # plt.plot(betas, costs, label="Final Cost vs Beta")
    plt.legend()
    plt.savefig(BASE_DIR + "final_cost_vs_beta.png")
    
if __name__ == "__main__":
    main()    
    