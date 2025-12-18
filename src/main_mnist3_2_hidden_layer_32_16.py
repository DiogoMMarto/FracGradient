"""
This script trains and evaluates a feedforward neural network on the HappyFace dataset 
with a 2 hidden layers of size [32,16] with only sigmoid activation functions. It supports multiple optimizers, 
including classical, adaptive, and several fractional variants.

Main features:
- Loads and preprocesses the HappyFace dataset (flatten images, one-hot encode labels).
- Defines a neural network architecture and cost function with L2 regularization.
- Runs training pipelines for different optimizers, saving results under `results/output_MNIST_3_2_hidden_layer_32_16/`.
- Supports grid search (`gen_grid_search`) to automatically explore optimizer hyperparameters.
- Can run pipelines sequentially or in parallel with ThreadPoolExecutor. 
  ThreadPoolExecutor has a perfomance overhead, so it should be used during hyperparameter search only.

To customize:
- Comment/uncomment entries in the `D` list to include/exclude specific optimizers.
- Modify `ARCHITECTURE` to change the hidden layer sizes (currently [32,16]).
- Adjust `NUM_EPOCHS`, `learning_rate`, `beta`, and other optimizer parameters for experiments.
- Switch `if False:` → `if True:` to enable parallel training with multiple threads. (line 117)
"""
from impl.Pipeline import Pipeline, gen_grid_search , end_pipeline_graphs
from impl.NN import NeuralNetwork
from impl.Optimizers import ClassicOptimizer , AdaptiveLearningRateOptimizer , MomentumOptimizer , FracOptimizer , FracOptimizer2 , AdamOptimizer , FracAdap , Frac3Optimizer, FracTrue , FracOptimizerBStable
from impl.CostFunctions import BinaryCrossEntropy , L2Regularization , ActivationFunction
from scipy.io import loadmat
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split

DATASET_PATH = "datasets/ex3data1.mat"
BASE_DIR = "results/output_MNIST_3_2_hidden_layer_32_16/"
NUM_EPOCHS = 3000
VERBOSE = False
ARCHITECTURE = [32,16]

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
            ARCHITECTURE, 
            400, 
            10, 
            BinaryCrossEntropy(
                regularization=L2Regularization(0.2),
                activation_function_names=[
                    "sigmoid",
                    "sigmoid",
                    "sigmoid"
                ]
            ), 
            Optimizer(**params)
        ),
        output,
        X_test=X_test,
        y_test=y_test
    )
    
    D = [
        ( ClassicOptimizer, {"learning_rate":1}, BASE_DIR + "classical/" , "SGD"),
        ( AdaptiveLearningRateOptimizer, {"initial_learning_rate":1}, BASE_DIR + "adaptive/" , "ALR"),
        # ( MomentumOptimizer, {"learning_rate":1, "momentum":0.5}, BASE_DIR + "momentum/"),
        # ( FracOptimizer, {"learning_rate":1}, BASE_DIR + "frac/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.1}, BASE_DIR + "fracB01/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.01}, BASE_DIR + "fracB001/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.001}, BASE_DIR + "fracB0001/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":10}, BASE_DIR + "fracB10/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.5}, BASE_DIR + "fracB05/" , "FracGradient V2"),
        # ( FracAdap, {"learning_rate":1,"beta":0.5}, BASE_DIR + "fracAdapB05/", "FracGradient V2 Adaptive"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.5}, BASE_DIR + "frac3B05/" , "Fractional Gradient Descent V3"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.05}, BASE_DIR + "frac3B005/"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.005}, BASE_DIR + "frac3B0005/"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":5}, BASE_DIR + "frac3B5/"),
        # ( FracOptimizer2, {"learning_rate":1}, BASE_DIR + "frac2/"),
        ( FracOptimizer2, {"learning_rate":1,"beta":1}, BASE_DIR + "frac2B0/", "FracGradient"),
        # ( FracOptimizer2, {"learning_rate":1,"beta":0.1}, BASE_DIR + "frac2B01/", "FracGradient"),
        # ( FracOptimizer2, {"learning_rate":1,"beta":5}, BASE_DIR + "frac2B5/"),
        # ( FracTrue, {"beta":0.5,"verbose":True}, BASE_DIR + "fracTrue/"),
        # ( AdamOptimizer, {"learning_rate":1}, BASE_DIR + "adam/"),
        # ( FracOptimizerBStable, {"learning_rate":1,"beta":0.5}, BASE_DIR + "fracBStable05/", "FracGradient B Stable"),
        # ( FracOptimizerBStable, {"learning_rate":1,"beta":0.01}, BASE_DIR + "fracBStable001/", "FracGradient B Stable"),
        # ( FracOptimizerBStable, {"learning_rate":1,"beta":0.01}, BASE_DIR + "fracBStable001/", "FracGradient B Stable 0.01"),
        # ( FracOptimizerBStable, {"learning_rate":1,"beta":0.005}, BASE_DIR + "fracBStable0005/", "FracGradient B Stable 0.005"),
        # ( FracOptimizerBStable, {"learning_rate":1,"beta":5}, BASE_DIR + "fracBStable5/", "FracGradient B Stable 5"),
    ]
    
    D2 = gen_grid_search([
        (ClassicOptimizer,{"learning_rate":[1,0.5,0.1,0.05,0.01,0.005,0.001]},BASE_DIR + "_classic", "Classic"),
        (FracOptimizer , {"learning_rate":[10,5,2,1,0.1,0.01],"beta":[5,1,0.5,0.1,0.05,0.01,0.005,0.001]}, BASE_DIR + "_frac_v2_/", "FracGradient V2"),
        (FracAdap , {"learning_rate":[1],"beta":list(2**np.arange(-10,3,0.3))}, BASE_DIR + "_frac_adap_v2/", "FracGradient V2 Adaptive"),
        (FracOptimizer , {"learning_rate":[1],"beta":list(2**np.arange(-10,3,0.3))}, BASE_DIR + "_frac_v2_/", "FracGradient V2")
    ]
        # [(FracOptimizer , {"learning_rate":[10,5,2,1,0.1,0.01,0.001],"beta":[5,1,0.5,0.1,0.05,0.01,0.005,0.001]}, BASE_DIR + "_frac_v2_/", "FracGradient V2"),
        #  (FracAdap , {"learning_rate":[5,2,1],"beta":[5,1,0.5,0.1,0.05,0.01]}, BASE_DIR + "_frac_adap_v2/", "FracGradient V2 Adaptive"),
        #  (FracOptimizer , {"learning_rate":[1],"beta":list(2**np.arange(-10,3,0.3))}, BASE_DIR + "_frac_v2_/", "FracGradient V2"),
        #  (FracAdap , {"learning_rate":[1],"beta":list(2**np.arange(-10,3,0.3))}, BASE_DIR + "_frac_adap_v2/", "FracGradient V2 Adaptive"),]
    )
    
    D.extend(D2)
    
    def run_pipeline(Optimizer,params,output):
        p = p_gen(Optimizer,params,output)
        p.run(epochs=NUM_EPOCHS,verbose=VERBOSE)
    
    if False:
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(run_pipeline, Optimizer,params,output) for Optimizer,params,output,_ in D]
            for future in futures:
                future.result()
    else:
        for Optimizer, params, output, name in D:
            p = p_gen(Optimizer, params, output)
            p.run(epochs=NUM_EPOCHS, verbose=VERBOSE)
    
    number_of_models_params = 0
    x_input_dim = X.shape[1]
    y_output_dim = y.shape[1]
    layers = [x_input_dim, *ARCHITECTURE, y_output_dim]
    print(f"Input dimension: {x_input_dim}, Output dimension: {y_output_dim}")
    for i,l in enumerate(layers[:-1]):
        previous = l + 1
        after = layers[i+1] 
        number_of_models_params += previous * after
    
    dataset_name = "MNIST"
    end_pipeline_graphs(D, BASE_DIR, number_of_models_params,dataset_name,"MNIST 3 - 2 hidden layer")
    
if __name__ == "__main__":
    main()    
    