import numpy as np
from impl.CostFunctions import sigmoid , BinaryCrossEntropy
from impl.Optimizers import ClassicOptimizer

def randInitializeWeights(L_in: int, L_out:int) -> np.ndarray:
    """
    Randomly initialize the weights of a layer with L_in incoming connections and L_out outgoing connections.
    The weights are initialized to small random values in the range [-epsilon, epsilon], where epsilon is calculated based on the number of connections.
    This helps in breaking symmetry and ensures that the neurons learn different features.
    
    Parameters
    ----------
    L_in : int
        Number of incoming connections to the layer.
    L_out : int
        Number of outgoing connections from the layer.
        
    Returns
    -------
    numpy.ndarray
        A numpy array of shape (L_out, L_in + 1) containing the initialized weights."""
    epi = (6**1/2) / (L_in + L_out)**1/2
    W = np.random.rand(L_out,L_in +1) *(2*epi) -epi
    return W

class NeuralNetwork:
    """
    A simple feedforward neural network implementation.
    This class allows for the creation of a neural network with a specified number of layers, input size, and output size.
    It supports forward propagation, weight initialization, and training using a specified cost function and optimizer.
    
    Parameters
    ----------
    layers : list of int
        A list containing the number of neurons in each hidden layer.
    input_size : int
        The number of features in the input data.
    output_size : int
        The number of output classes (for classification tasks).
    cost_function : BinaryCrossEntropy
        An instance of a cost function class that computes the cost and gradients.
    optimizer : ClassicOptimizer
        An instance of an optimizer class that updates the weights during training.
        
    Attributes
    ----------
    layers : list of int
        The number of neurons in each hidden layer.
    input_size : int
        The number of features in the input data.
    output_size : int
        The number of output classes.
    cost_function : BinaryCrossEntropy
        The cost function used for training the neural network.
    optimizer : ClassicOptimizer
        The optimizer used to update the weights during training.
    weights : list of numpy arrays
        The weights of the neural network, initialized randomly.
    X : numpy.ndarray
        The input data used for training.
    y : numpy.ndarray
        The target output data used for training.
        
    Methods
    -------
    initialize_weights(layers: list[int]) -> list[np.ndarray]
        Initializes the weights of the neural network based on the specified layer sizes.
    forward_propagation(X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]
        Performs forward propagation through the network, computing activations and pre-activations for each layer.
    forward_propagation_weigths(weights: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]
        Performs forward propagation using a specified set of weights, returning activations and pre-activations.
    predict(X: np.ndarray) -> np.ndarray
        Makes predictions on the input data by performing forward propagation through the network.
    fit(X: np.ndarray, y: np.ndarray, epochs: int = 100, verbose: bool = False)
        Trains the neural network using the provided input data and target output data for a specified number of epochs.
    """
    def __init__(self, layers: list[int] , input_size: int , output_size: int, cost_function: BinaryCrossEntropy, optimizer: ClassicOptimizer):
        self.layers = layers
        self.input_size = input_size
        self.output_size = output_size
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.weights = self.initialize_weights(layers)
        self.optimizer.parent = self # type: ignore
    
    def initialize_weights(self, layers: list[int]) -> list[np.ndarray]:
        """
        Initializes the weights of the neural network based on the specified layer sizes.
        
        Parameters
        ----------
        layers : list of int
            A list containing the number of neurons in each hidden layer.
            
        Returns
            -------
            list of numpy arrays
                A list of weight matrices for each layer, where each matrix has shape (number of neurons in the next layer, number of neurons in the current layer + 1).
        """
        weights = []
        layer_sizes = [self.input_size] + layers + [self.output_size]
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            weight_matrix = randInitializeWeights(layer_sizes[i], layer_sizes[i + 1])
            weights.append(weight_matrix)
        return weights
    
    def forward_propagation_self(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Performs forward propagation through the neural network.
        This method computes the activations and pre-activations for each layer in the network.
        
        Parameters
        ----------
        X : numpy.ndarray
            The input data for the forward propagation, with shape (number of examples, number of features).
            
        Returns
        -------
        tuple of lists
            A list of activations for each layer (including the input layer) and a list of pre-activations (Z values) for each layer.
            Each activation is a numpy array with shape (number of examples, number of neurons in the layer).
        """
        X_p_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        A_ = [X_p_bias]
        Z_ = [X_p_bias]
        for i in range(len(self.layers)+1):
            Z = A_[-1] @ self.weights[i].T
            if i < len(self.layers):
                Z = np.hstack((np.ones((Z.shape[0], 1)), Z))
            A = sigmoid(Z)
            A_.append(A)
            Z_.append(Z)
        return A_, Z_
    
    def forward_propagation(self, weights: list[np.ndarray], X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Performs forward propagation through the neural network using a specified set of weights.
        This method computes the activations and pre-activations for each layer in the network using the provided weights.
        
        Parameters
        ----------
        weights : list of numpy arrays
            A list of weight matrices for each layer, where each matrix has shape (number of neurons in the next layer, number of neurons in the current layer + 1).
        X : numpy.ndarray
            The input data for the forward propagation, with shape (number of examples, number of features).
            
        Returns
        -------
        tuple of lists
            A list of activations for each layer (including the input layer) and a list of pre-activations (Z values) for each layer.
            Each activation is a numpy array with shape (number of examples, number of neurons in the layer).
        """
        X_p_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        A_ = [X_p_bias]
        Z_ = [X_p_bias]
        for i in range(len(self.layers)+1):
            Z = A_[-1] @ weights[i].T
            if i < len(self.layers):
                Z = np.hstack((np.ones((Z.shape[0], 1)), Z))
            A = sigmoid(Z)
            A_.append(A)
            Z_.append(Z)
        return A_, Z_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the input data by performing forward propagation through the network.
        This method computes the final output of the network for the given input data.
        
        Parameters
        ----------
        X : numpy.ndarray
            The input data for the prediction, with shape (number of examples, number of features).
            
        Returns
        -------
        numpy.ndarray
            The predicted output of the neural network, with shape (number of examples, number of output classes).
        """
        A = X
        for i in range(len(self.layers)+1):
            A = np.hstack((np.ones((A.shape[0], 1)), A))
            Z = A @ self.weights[i].T
            A = sigmoid(Z)
        return A
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int=100, verbose: bool=False):
        """
        Trains the neural network using the provided input data and target output data for a specified number of epochs.
        This method performs forward propagation, computes the cost, calculates gradients, and updates the weights using the specified optimizer.
        
        Parameters
        ----------
        X : numpy.ndarray
            The input data for training, with shape (number of examples, number of features).
        y : numpy.ndarray
            The target output data for training, with shape (number of examples, number of output classes).
        epochs : int, optional
            The number of epochs to train the neural network. Default is 100.
        verbose : bool, optional
            If True, prints detailed information during training. Default is False.
        """
        self.X = X
        self.y = y
        self.optimizer.verbose = verbose
        self.optimizer.reset()
        for _ in range(epochs):
            A_ , Z_ = self.forward_propagation_self(X)
            cost = self.cost_function.cost(A_, self.weights,y)
            grads = self.cost_function.gradient(A_,Z_, self.weights, y)
            self.optimizer.step(self.weights, grads, cost)