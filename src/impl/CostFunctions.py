import numpy as np
from scipy.special import gamma

def sigmoid(z):
    return 1 / (1 + np.exp(-z)) 

def sigmoid_gradient(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

class L2Regularization:
    def __init__(self, lambda_=0.01):
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
        lambda_ : float, optional
            The regularization parameter (default is 0.01).

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
    
    def __init__(self, activation_function=sigmoid, activation_gradient=sigmoid_gradient, regularization: L2Regularization=None):
        self.activation_function = activation_function
        self.activation_gradient = activation_gradient
        self.regularization = regularization
    
    def cost(self,A: list[np.ndarray], weigths: list[np.ndarray], y: np.ndarray) -> float:        
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
    
def frac_gradient_from_gradient(fraction: float, gradient: list[np.ndarray], weights: list[np.ndarray], prev_weights: list[np.ndarray]) -> list[np.ndarray]:
    """
    Compute the fractional gradients from the given gradients.

    Parameters
    ----------
    fraction : float
        The fractional order (0 < fraction <= 1).
    gradients : list of numpy arrays
        The gradients of the neural network.
    Returns
    -------
    list of numpy arrays
        The fractional gradients.
    """
    return gradient * (np.abs(weights - prev_weights) ** (1 - fraction)) / gamma(2 - fraction)