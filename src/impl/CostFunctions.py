from enum import Enum
import numpy as np
from scipy.special import gamma

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.
    
    Parameters
    ----------
    z : numpy array
        The input to the sigmoid function.
        
    Returns
    -------
    numpy array
        The output of the sigmoid function.
    """
    return 1 / (1 + np.exp(-z)) 

def sigmoid_gradient(z: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the sigmoid function.
    
    Parameters
    ----------
    z : numpy array
        The input to the sigmoid function.
    
    Returns
    -------
    numpy array
        The gradient of the sigmoid function.
    """
    sig = sigmoid(z)
    return sig * (1 - sig)

def relu(z: np.ndarray) -> np.ndarray:
    """
    Compute the ReLU activation function.

    Parameters
    ----------
    z : numpy array
        The input to the ReLU function.

    Returns
    -------
    numpy array
        The output of the ReLU function.
    """
    return np.maximum(0, z)

def relu_gradient(z: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the ReLU function.

    Parameters
    ----------
    z : numpy array
        The input to the ReLU function.

    Returns
    -------
    numpy array
        The gradient of the ReLU function.
    """
    return np.where(z > 0, 1, 0)

def tanh(z: np.ndarray) -> np.ndarray:
    """
    Compute the hyperbolic tangent activation function.

    Parameters
    ----------
    z : numpy array
        The input to the tanh function.

    Returns
    -------
    numpy array
        The output of the tanh function.
    """
    return np.tanh(z)

def tanh_gradient(z: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the hyperbolic tangent function.

    Parameters
    ----------
    z : numpy array
        The input to the tanh function.

    Returns
    -------
    numpy array
        The gradient of the tanh function.
    """
    return 1 - np.tanh(z) ** 2

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Compute the softmax activation function.

    Parameters
    ----------
    z : numpy array
        The input to the softmax function.

    Returns
    -------
    numpy array
        The output of the softmax function.
    """
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def softmax_gradient(z: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the softmax function.

    Parameters
    ----------
    z : numpy array
        The input to the softmax function.

    Returns
    -------
    numpy array
        The gradient of the softmax function.
    """ 
    # return error unimplemented, as softmax gradient is typically handled differently in backpropagation
    raise NotImplementedError("Softmax gradient is not implemented. It is typically handled differently in backpropagation.")
class ActivationFunction(Enum):
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"
    SOFTMAX = "softmax"
 
ACTIVATION_MAP = {
    ActivationFunction.SIGMOID.value: (sigmoid, sigmoid_gradient),
    ActivationFunction.RELU.value: (relu, relu_gradient),
    ActivationFunction.TANH.value: (tanh, tanh_gradient),
    ActivationFunction.SOFTMAX.value: (softmax, softmax_gradient)
} 
    
def get_activation_function(name: str) -> tuple[callable, callable]:
    """
    Get the activation function and its gradient based on the name.

    Parameters
    ----------
    name : str
        The name of the activation function.

    Returns
    -------
    tuple of callable
        The activation function and its gradient.
    
    Raises
    ------
    ValueError
        If the activation function name is not recognized.
    """
    if name in ACTIVATION_MAP:
        return ACTIVATION_MAP[name]
    else:
        raise ValueError(f"Unknown activation function: {name}")

class L2Regularization:
    """
    L2 Regularization for neural networks.
    This class computes the L2 regularization term and its gradient for the weights of a neural network.
    
    Parameters
    ----------
    lambda : float, optional
        The regularization parameter (default is 0.01).
    """
    def __init__(self, lambda_: float=0.01):
        self.lambda_ = lambda_
    
    def cost(self, weights: list[np.ndarray] , num_examples: int) -> float:
        """
        Compute the L2 regularization term for the given weights.

        Parameters
        ----------
        weights : list of numpy arrays
            The weights of the neural network.
        num_examples : int
            The number of examples in the training set.

        Returns
        -------
        float
            The L2 regularization term.
        """
        reg_term = 0.0
        for w in weights:
            reg_term += np.sum(np.square(w[:, 1:]))  # Exclude bias terms
        return self.lambda_ / (2 * num_examples) * reg_term
    
    def grad(self, weights: list[np.ndarray], num_examples: int) -> list[np.ndarray]:
        """
        Compute the gradient of the L2 regularization term for the given weights.

        Parameters
        ----------
        weights : list of numpy arrays
            The weights of the neural network.
        num_examples : int
            The number of examples in the training set.

        Returns
        -------
        list of numpy arrays
            The gradient of the L2 regularization term.
        """
        grads = []
        for w in weights:
            grad = (self.lambda_/num_examples) * np.hstack((np.zeros((w.shape[0],1)),w[:,1:]))
            grads.append(grad)
        return grads

class BinaryCrossEntropy:
    """
    Binary Cross Entropy cost function for neural networks.
    This class computes the cost and gradient for a binary classification task using the sigmoid activation function.
    It can also include L2 regularization if specified.
    
    Parameters
    ----------
    activation_function : callable, optional
        The activation function to use (default is sigmoid).
    activation_gradient : callable, optional
        The gradient of the activation function (default is sigmoid_gradient).
    regularization : L2Regularization, optional
        An instance of L2Regularization for regularization (default is None).
    """
    def __init__(self, activation_function_names: list[str] = [], regularization: L2Regularization | None=None):
        self._activation_str = activation_function_names
        _activations , _gradients_activation = self._init_activations()
        self.activation_function =  _activations
        self.activation_gradient =  _gradients_activation
        self.regularization = regularization
        
    def _init_activations(self) -> tuple[list[callable], list[callable]]:
        ret = ([], [])
        for name in self._activation_str:
            _activation, _activation_gradient = get_activation_function(name)
            ret[0].append(_activation)
            ret[1].append(_activation_gradient)
        return ret
        
    def validate_activation(self,num):    
        """
        Validate the number of activation functions.

        Parameters
        ----------
        num : int
            The number of activation functions provided.

        Returns
        -------
        bool
            True if the number of activation functions matches the expected count.

        Raises
        ------
        ValueError
            If the number of activation functions does not match the expected count when using a list of activation functions.
        """
        if len(self._activation_str) != num:
            raise ValueError(f"Expected {num} activation functions, but got {len(self._activation_str)}")
        return True
        
    def get_activation(self,index):
        """
        Get the activation function for a given layer.

        Parameters
        ----------
        index : int
            The index of the layer.

        Returns
        -------
        callable
            The activation function for the layer.
        """
        return self.activation_function[index]
    
    def _get_activation_gradient(self,index):
        return self.activation_gradient[index]
    
    def cost(self,A: list[np.ndarray], weigths: list[np.ndarray], y: np.ndarray) -> float:      
        """
        Compute the cost for the binary cross entropy loss function.
        
        Parameters
        ----------
        A : list of numpy arrays
            The activations of the neural network layers.
        weigths : list of numpy arrays
            The weights of the neural network.
        y : numpy array
            The true labels for the training examples, one-hot encoded.
            
        Returns
        -------
        float
            The computed cost.
        """  
        num_examples , num_labels = y.shape
        J = 0
        P = A[-1]
        for j in range(num_labels):
            J = J + sum(-y[:,j] * np.log(P[:,j]) - (1-y[:,j])*np.log(1-P[:,j]))
        J /= num_examples
        if self.regularization is not None:
            reg_term = self.regularization.cost(weigths, num_examples)
            J += reg_term
        return J
    
    def gradient(self,A_: list[np.ndarray], Z_: list[np.ndarray], weigths: list[np.ndarray], y: np.ndarray) -> list[np.ndarray]:
        """
        Compute the gradient of the cost function with respect to the weights.
        
        Parameters
        ----------
        A : list of numpy arrays
            The activations of the neural network layers.
        Z : list of numpy arrays
            The pre-activations of the neural network layers.
        weigths : list of numpy arrays
            The weights of the neural network.
        y : numpy array
            The true labels for the training examples, one-hot encoded.
            
        Returns
        -------
        list of numpy arrays
            The gradients of the cost function with respect to the weights.
        """
        m = y.shape[0]
        
        deltas = [A_[-1] - y]
        for i in range(len(weigths) - 1, 0, -1):
            Z = Z_[i]
            delta = deltas[-1] @ weigths[i] *  self._get_activation_gradient(i)(Z)  # sigmoid_gradient(Z)
            delta = delta[:, 1:]
            deltas.append(delta)

        deltas.reverse()
        
        if self.regularization is not None:
            regs = self.regularization.grad(weigths, m)
            grads = []
            for i in range(len(weigths)):
                grad = (deltas[i].T @ A_[i]) / m
                grad += regs[i]
                grads.append(grad)
        else:
            grads = [(deltas[i].T @ A_[i]) / m for i in range(len(weigths))]
        return grads
    
def frac_gradient_from_gradient(fraction: float, gradient: np.ndarray, weights: np.ndarray, prev_weights: np.ndarray) -> np.ndarray:
    """
    Compute the fractional gradient from the standard gradient.
    This function applies a fractional power to the difference between the current and previous weights,
    scaled by the gradient and a gamma function.

    Parameters
    ----------
    fraction : float
        The fractional exponent to apply to the difference between current and previous weights.
    gradient : numpy arrays
        The standard gradient computed from the cost function.
    weights : numpy arrays
        The current weights of the neural network.
    prev_weights : numpy arrays
        The previous weights of the neural network.
        
    Returns
    -------
    numpy arrays
        The fractional gradient, which is the standard gradient scaled by the difference between current and previous weights,
        raised to the power of (1 - fraction) and divided by gamma(2 - fraction).
    """
    return gradient * (np.abs(weights - prev_weights) ** (1 - fraction)) / gamma(2 - fraction) # type: ignore