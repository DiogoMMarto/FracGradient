import numpy as np
from impl.CostFunctions import sigmoid , BinaryCrossEntropy
from impl.Optimizers import ClassicOptimizer

def randInitializeWeights(L_in, L_out):
    epi = (6**1/2) / (L_in + L_out)**1/2
    W = np.random.rand(L_out,L_in +1) *(2*epi) -epi
    return W

class NeuralNetwork:
    def __init__(self, layers , input_size , output_size, cost_function: BinaryCrossEntropy, optimizer: ClassicOptimizer):
        self.layers = layers
        self.input_size = input_size
        self.output_size = output_size
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.weights = self.initialize_weights(layers)
    
    def initialize_weights(self, layers):
        weights = []
        layer_sizes = [self.input_size] + layers + [self.output_size]
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            weight_matrix = randInitializeWeights(layer_sizes[i], layer_sizes[i + 1])
            weights.append(weight_matrix)
        return weights
    
    def forward_propagation(self, X):
        X_p_bias = np.hstack((X, np.ones((X.shape[0], 1))))
        A_ = [X_p_bias]
        A = X_p_bias
        for i in range(len(self.layers)+1):
            Z = A @ self.weights[i].T
            A = sigmoid(Z)
            if i < len(self.layers):
                A = np.hstack((A, np.ones((A.shape[0], 1))))
            A_.append(A)
        return A_

    def predict(self, X):
        A = X
        for i in range(len(self.layers)+1):
            A = np.hstack((A, np.ones((A.shape[0], 1))))
            Z = A @ self.weights[i].T
            A = sigmoid(Z)
        return A
    
    def fit(self, X, y, epochs=100, verbose=False):
        self.optimizer.verbose = verbose
        self.optimizer.reset()
        for _ in range(epochs):
            A_ = self.forward_propagation(X)
            cost = self.cost_function.cost(A_, self.weights,y)
            grads = self.cost_function.gradient(A_, self.weights, y)
            self.optimizer.step(self.weights, grads, cost)