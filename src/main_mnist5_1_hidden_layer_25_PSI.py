"""
This script trains and evaluates a feedforward neural network on the MNIST dataset 
with a single hidden layer of size 25. It supports multiple optimizers, 
including classical, adaptive, and several fractional variants.

Main features:
- Loads and preprocesses the MNIST dataset (flatten images, one-hot encode labels).
- Defines a neural network architecture and cost function with L2 regularization.
- Runs training pipelines for different optimizers, saving results under `results/output_MNIST_1_1_hidden_layer_25/`.
- Supports grid search (`gen_grid_search`) to automatically explore optimizer hyperparameters.
- Can run pipelines sequentially or in parallel with ThreadPoolExecutor. 
  ThreadPoolExecutor has a perfomance overhead, so it should be used during hyperparameter search only.

To customize:
- Comment/uncomment entries in the `D` list to include/exclude specific optimizers.
- Modify `ARCHITECTURE` to change the hidden layer sizes (currently [25]).
- Adjust `NUM_EPOCHS`, `learning_rate`, `beta`, and other optimizer parameters for experiments.
- Switch `if False:` → `if True:` to enable parallel training with multiple threads. (line 111)
"""
from impl.Pipeline import Pipeline, gen_grid_search , end_pipeline_graphs
from impl.NN import NeuralNetwork
from impl.Optimizers import ClassicOptimizer , AdaptiveLearningRateOptimizer , FracOptimizer , FracOptimizer2 , FracAdap , FracOptimizerPsi
from impl.CostFunctions import BinaryCrossEntropy , L2Regularization
from scipy.io import loadmat
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split

DATASET_PATH = "datasets/ex3data1.mat"
BASE_DIR = "results/" + "main_mnist5_1_hidden_layer_25_PSI" + "/"
NUM_EPOCHS = 3000
VERBOSE = False
ARCHITECTURE = [25]


class psi_gen_power:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return np.power(x, self.n)
    
    def __repr__(self):
        return f"power_{self.n}_"
    
    def __str__(self):
        return f"power_{self.n}_"
    
    def to_dict(self):
        return {"type": "psi_gen_power", "n": self.n}
    
class psi_gen_xex:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return x * np.exp(x * self.n)
    
    def __repr__(self):
        return f"xex_{self.n}_"
    
    def __str__(self):
        return f"xex_{self.n}_"
    
    def to_dict(self):
        return {"type": "psi_gen_xex", "n": self.n}

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
                ]
            ), 
            Optimizer(**params)
        ),
        output,
        X_test=X_test,
        y_test=y_test
    )
    
    D = [
        ( ClassicOptimizer, {"learning_rate":2}, BASE_DIR + "classical_2/" , "Gradient Descent"),
        ( AdaptiveLearningRateOptimizer, {"initial_learning_rate":1}, BASE_DIR + "adaptive/" , "Adaptive Learning Rate"),

        ( FracOptimizer, {"learning_rate":1,"beta":0.5}, BASE_DIR + "fracB05/" , "FracGradient V2"),
        ( FracAdap, {"learning_rate":1,"beta":0.5}, BASE_DIR + "fracAdapB05/", "FracGradient V2 Adaptive"),
        # ( FracOptimizer2, {"learning_rate":1.5,"beta":0.5}, BASE_DIR + "frac2B01/", "FracGradient"),
    ]
    
    D2 = gen_grid_search(
        [
         (FracOptimizer , {"learning_rate":[1,1.5,2],"beta":list(2**np.arange(-6,2.1,0.3))}, BASE_DIR + "_frac_v2_/", "FracGradient V2"),
         (FracAdap , {"learning_rate":[1],"beta":list(2**np.arange(-6,2.1,0.3))}, BASE_DIR + "_frac_adap_v2/", "FracGradient V2 Adaptive"),
         (FracOptimizerPsi , {"learning_rate":[0.5,0.1],"beta":list(2**np.arange(-5,2.1,0.5)),"psi":[psi_gen_power(n) for n in [2/3,4/5,1,6/5,4/3,2]]}, BASE_DIR + "_frac_psi/", "FracGradient Psi"),
         (FracOptimizerPsi , {"learning_rate":[0.05,0.01],"beta":list(2**np.arange(-3,0,0.5)),"psi":[psi_gen_power(n) for n in [2/3,4/5,1,6/5,4/3,2]]}, BASE_DIR + "_frac_psi/", "FracGradient Psi"),
         (FracOptimizerPsi , {"learning_rate":[0.1,0.05,0.01,0.5,1],"beta":list(2**np.arange(-3,0.1,0.5)),"psi":[psi_gen_xex(n) for n in [-1,-0.5,-0.25,-0.1,0.1,0.25,0.5,1]]}, BASE_DIR + "_frac_psi/", "FracGradient Psi"),
         (FracOptimizerPsi , {"learning_rate":[1],"beta":list(2**np.arange(-3,-2,0.5)),"psi":[psi_gen_power(n) for n in [1,1.01,1.1]]}, BASE_DIR + "_frac_psi/", "FracGradient Psi"),
         (FracOptimizerPsi , {"learning_rate":[1],"beta":list(2**np.arange(-3,-2,0.5)),"psi":[psi_gen_power(n) for n in [1]]}, BASE_DIR + "_frac_psi/", "FracGradient Psi"),
         
        ]
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
    
    end_pipeline_graphs(D, BASE_DIR, number_of_models_params, "MNIST","MNIST 1 - 1 hidden layer")
    
if __name__ == "__main__":
    main()    
    