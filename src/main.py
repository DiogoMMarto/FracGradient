from impl.Pipeline import Pipeline
from impl.NN import NeuralNetwork
from impl.Optimizers import ClassicOptimizer , AdaptiveLearningRateOptimizer , MomentumOptimizer , FracOptimizer , FracOptimizer2 , FOMA
from impl.CostFunctions import BinaryCrossEntropy , L2Regularization
from scipy.io import loadmat
import numpy as np

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
    
    # conver y to one hot enconded
    
    p = Pipeline(
        X, 
        y, 
        NeuralNetwork(
            [25], 
            400, 
            10, 
            BinaryCrossEntropy(
                regularization=L2Regularization(0.2)
            ), 
            FOMA(learning_rate=1,beta=1)
        ),
        "output/foma/"
    )
    p.run(epochs=500, verbose=True)
    

if __name__ == "__main__":
    main()    
    