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
    def __init__(self, activation_function=sigmoid, activation_gradient=sigmoid_gradient, regularization: L2Regularization | None=None):
        self.activation_function = activation_function
        self.activation_gradient = activation_gradient
        self.regularization = regularization
    
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
            A = Z_[i]
            delta = deltas[-1] @ weigths[i] * sigmoid_gradient(A)
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