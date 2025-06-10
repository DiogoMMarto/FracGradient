from time import time
import numpy as np
from impl.CostFunctions import frac_gradient_from_gradient
from scipy.optimize import minimize_scalar

class ClassicOptimizer:
    """
    A base class for classic optimization algorithms used in training neural networks.
    This class provides a framework for implementing various optimization techniques such as gradient descent, momentum, and adaptive learning rates.
    It tracks the cost and time history of the optimization process and provides methods for stepping through the optimization process, resetting the optimizer, and retrieving the history.
    It can be extended to implement specific optimization algorithms.
    
    Parameters
    ----------
    learning_rate : float, optional
        The learning rate for the optimizer. Default is 0.01.
    verbose : bool, optional
        If True, enables verbose output during the optimization process. Default is False.
        
    Methods
    -------
    step(params, grads, cost):
        Performs a single optimization step using the provided parameters, gradients, and cost.
    _end_step(cost):
        Finalizes the step by updating the cost and time history, and increments the iteration counter.
    _print_step_info(cost):
        Prints the step information including cost and time if verbose is enabled.
    reset():
        Resets the optimizer's history and iteration counter.
    get_history():
        Returns the history of costs and times recorded during the optimization process.
    
    Attributes:
        learning_rate (float): The learning rate for the optimizer.
        history (dict): A dictionary to store the cost and time history.
        i (int): Counter for iterations.
        base_time (float or None): Base time for tracking elapsed time.
    """
    def __init__(self, learning_rate: float=0.01, verbose: bool=False):
        """
        Initializes the ClassicOptimizer with a specified learning rate and verbosity.

        Args:
            learning_rate (float): The learning rate for the optimizer. Default is 0.01.
            verbose (bool): If True, enables verbose output. Default is False.
        """
        self.learning_rate = learning_rate
        self.history = {}
        self.history['cost'] = []
        self.history['time'] = []
        self.i = 0
        self.base_time = None
        self.verbose = verbose
        self.parent = None  # Placeholder for parent object, if needed later

    def step(self, params: list[np.ndarray], grads:list[np.ndarray], cost: float):
        
        for i in range(len(params)):
            params[i] -= self.learning_rate * grads[i]
        self._end_step(cost)
        
    def _end_step(self,cost:float):
        self.history['cost'].append(cost)
        if self.base_time is None:
            self.base_time = time()
        self.history['time'].append(time() - self.base_time)
        self.i += 1
        if self.verbose:
            self._print_step_info(cost)
    
    def _print_step_info(self, cost:float):
        print(f"Step {self.i}: Cost = {cost:.4f}, Time = {self.history['time'][-1]:.4f} seconds")
        
    def reset(self):
        self.history = {}
        self.history['cost'] = []
        self.history['time'] = []
        self.i = 0
        
    def get_history(self) -> dict:
        return self.history
    
class MomentumOptimizer(ClassicOptimizer):
    """
    A class implementing the Momentum optimization algorithm, which accelerates gradient descent by accumulating a velocity vector in the direction of the gradients.
    This class extends the ClassicOptimizer and provides methods to perform optimization steps with momentum, reset the optimizer, and retrieve the optimization history.

    Parameters
    ----------
    learning_rate : float, optional
        The initial learning rate for the optimizer. Default is 0.01.
    momentum : float, optional
        The momentum factor, which determines the contribution of the previous velocity to the current update. Default is 0.9.
    verbose : bool, optional
        If True, enables verbose output during the optimization process. Default is False.
    
    Methods
    -------
    step(params, grads, cost):
        Performs a single optimization step using the provided parameters, gradients, and cost, applying momentum to the updates.
    reset():
        Resets the optimizer's velocity and history.
    get_history():
        Returns the history of costs and times recorded during the optimization process.
        
    Attributes:
        momentum (float): The momentum factor for the optimizer.
        velocity (list or None): A list to store the velocity for each parameter, initialized to None.
        initial_learning_rate (float): The initial learning rate for the optimizer.
    """

    def __init__(self, learning_rate=0.01, momentum=0.9, verbose=False):
        """
        Initialize the MomentumOptimizer with specified learning rate and momentum.

        Args:
            learning_rate (float): The initial learning rate for the optimizer. Default is 0.01.
            momentum (float): The momentum factor. Default is 0.9.
            verbose (bool): If True, enables verbose output. Default is False.

        Attributes:
            momentum (float): Stores the momentum factor.
            velocity (list or None): Stores the velocity for each parameter, initialized to None.
            initial_learning_rate (float): Stores the initial learning rate.
        """
        super().__init__(learning_rate, verbose)
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads, cost):
        if self.velocity is None:
            self.velocity = [np.zeros_like(param) for param in params]
        
        for i in range(len(params)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grads[i]
            params[i] += self.velocity[i]
        
        super().step(params, grads, cost)
        
    def reset(self):
        super().reset()
        self.velocity = None
        
class AdaptiveLearningRateOptimizer(ClassicOptimizer):
    """
    A class implementing an adaptive learning rate optimizer that adjusts the learning rate based on the cost function.
    This class extends the ClassicOptimizer and provides methods to perform optimization steps with adaptive learning rates, reset the optimizer, and retrieve the optimization history.
    
    Parameters
    ----------
    initial_learning_rate : float, optional
        The initial learning rate for the optimizer. Default is 0.01.
    increase_rate : float, optional
        The rate to increase the learning rate when the cost decreases. Default is 1.1.
    decay_rate : float, optional
        The rate to decay the learning rate when the cost increases. Default is 0.6.
    verbose : bool, optional
        If True, enables verbose output during the optimization process. Default is False.
    
    Methods
    -------
    step(params, grads, cost):
        Performs a single optimization step using the provided parameters, gradients, and cost, adjusting the learning rate based on the cost.
    compute_learning_rate(cost):
        Computes the new learning rate based on the current cost and the history of costs.
    reset():
        Resets the optimizer's history and learning rate to the initial value.
    get_history():
        Returns the history of costs and times recorded during the optimization process.
        
    Attributes:
        increase_rate (float): The rate to increase the learning rate when the cost decreases.
        decay_rate (float): The rate to decay the learning rate when the cost increases.
        initial_learning_rate (float): The initial learning rate for the optimizer.
        
    """
    def __init__(self, initial_learning_rate=0.01, increase_rate=1.1, decay_rate=0.6, verbose=False):
        """
        Initialize the AdaptiveLearningRateOptimizer with specified initial learning rate, increase rate and decay rate.

        Args:
            initial_learning_rate (float): The initial learning rate for the optimizer. Default is 0.01.
            increase_rate (float): The rate to increase the learning rate when the cost decreases. Default is 1.1.
            decay_rate (float): The rate to decay the learning rate when the cost increases. Default is 0.6.
            verbose (bool): If True, enables verbose output. Default is False.
        """
        super().__init__(initial_learning_rate, verbose)
        self.increase_rate = increase_rate
        self.decay_rate = decay_rate
        self.initial_learning_rate = initial_learning_rate

    def step(self, params, grads, cost):
        self.compute_learning_rate(cost)
        super().step(params, grads, cost)
        
    def compute_learning_rate(self, cost):
        if self.i == 0:
            self.learning_rate = self.initial_learning_rate
        elif cost < self.history['cost'][-1]:
            self.learning_rate *= self.increase_rate
        else:
            self.learning_rate *= self.decay_rate
        return self.learning_rate  
        
    def reset(self):
        super().reset()
        self.learning_rate = self.initial_learning_rate
        
class MomentumAdaptiveOptimizer(MomentumOptimizer, AdaptiveLearningRateOptimizer):
    def __init__(self, initial_learning_rate=0.01, momentum=0.9, increase_rate=1.1, decay_rate=0.6, verbose=False):
        """
        Initialize the MomentumAdaptiveOptimizer with specified initial learning rate, momentum, increase rate and decay rate.

        Args:
            initial_learning_rate (float): The initial learning rate for the optimizer. Default is 0.01.
            momentum (float): The momentum factor. Default is 0.9.
            increase_rate (float): The rate to increase the learning rate when the cost decreases. Default is 1.1.
            decay_rate (float): The rate to decay the learning rate when the cost increases. Default is 0.6.
            verbose (bool): If True, enables verbose output. Default is False.

        Attributes:
            momentum (float): Stores the momentum factor.
            increase_rate (float): Stores the increase rate.
            decay_rate (float): Stores the decay rate.
            initial_learning_rate (float): Stores the initial learning rate.
        """
        MomentumOptimizer.__init__(self, initial_learning_rate, momentum, verbose)
        AdaptiveLearningRateOptimizer.__init__(self, initial_learning_rate, increase_rate, decay_rate, verbose)

    def step(self, params, grads, cost):
        self.compute_learning_rate(cost)
        MomentumOptimizer.step(self, params, grads, cost)

class AdamOptimizer(ClassicOptimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, verbose=False):
        """
        Initialize the AdamOptimizer with specified learning rate, beta1, beta2, epsilon and verbose.

        Args:
            learning_rate (float): The initial learning rate for the optimizer. Default is 0.001.
            beta1 (float): The first moment estimation factor. Default is 0.9.
            beta2 (float): The second moment estimation factor. Default is 0.999.
            epsilon (float): A small value added to the variance for numerical stability. Default is 1e-8.
            verbose (bool): If True, enables verbose output. Default is False.

        Attributes:
            beta1 (float): Stores the first moment estimation factor.
            beta2 (float): Stores the second moment estimation factor.
            epsilon (float): Stores the small value to add for numerical stability.
            initial_learning_rate (float): Stores the initial learning rate.
            m (list or None): Stores the first moment of the gradient for each parameter, initialized to None.
            v (list or None): Stores the second moment of the gradient for each parameter, initialized to None.
        """
        super().__init__(learning_rate, verbose)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.initial_learning_rate = learning_rate

    def step(self, params, grads, cost):
        if self.m is None or self.v is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]
        
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** (self.i + 1))
            v_hat = self.v[i] / (1 - self.beta2 ** (self.i + 1))
            
            params[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        self._end_step(cost)
        
    def reset(self):
        super().reset()
        self.m = None
        self.v = None
        self.learning_rate = self.initial_learning_rate
     
def alpha_function(norm_GradCost,beta):
    return 1 - 2 / np.pi * np.arctan(norm_GradCost * beta)
    
class FracOptimizer(ClassicOptimizer):
    def __init__(self, learning_rate=0.001, alpha_func = alpha_function, beta=0.9,  verbose=False):
        """
        Initialize the FracOptimizer with specified learning rate, and verbosity.

        Args:
            learning_rate (float): The initial learning rate for the optimizer. Default is 0.001.
            verbose (bool): If True, enables verbose output. Default is False.

        Attributes:
            learning_rate (float): Stores the initial learning rate.
            fraction (float): Stores the fractional order.
        """
        super().__init__(learning_rate, verbose)
        self.alpha_func = alpha_func
        self.beta = beta
        self.previous_weigths = None
        self.previous_grads = None
        self.previous_cost = None   

    def step(self, params, grads, cost):
        if self.previous_grads is None or self.previous_cost is None or self.previous_weigths is None:
            new_grads = grads
        else:
            new_grads = []
            for i in range(len(params)):
                norm_grad = np.linalg.norm(self.previous_grads[i])
                alpha = self.alpha_func(norm_grad, self.beta)
                new_grad = frac_gradient_from_gradient(alpha, self.previous_grads[i], params[i], self.previous_weigths[i])
                new_grads.append(new_grad)

        self.previous_grads = [grad.copy() for grad in grads]
        self.previous_cost = cost
        self.previous_weigths = [weigths.copy() for weigths in params]
        super().step(params, new_grads, cost)


    def reset(self):
        super().reset()
        self.previous_grads = None
        self.previous_cost = None
        self.previous_weigths = None
        
class FracOptimizer2(ClassicOptimizer):

    def __init__(self, learning_rate=0.01, alpha_function = alpha_function, beta=0.9, verbose=False):
        super().__init__(learning_rate, verbose)
        self.alpha_func = alpha_function
        self.beta = 0.9
        self.previous_weigths = None
        self.previous_grads = None
        self.previous_cost = None
    
    def step(self, params, grads, cost):
        if self.previous_grads is None or self.previous_cost is None or self.previous_weigths is None:
            new_grads = grads
        else:
            new_grads = []
            for i in range(len(params)):
                norm_grad = np.abs(self.previous_grads[i])
                alpha = self.alpha_func(norm_grad, self.beta)
                new_grad = frac_gradient_from_gradient(alpha, self.previous_grads[i], params[i], self.previous_weigths[i])
                new_grads.append(new_grad)
            
        self.previous_grads = [grad.copy() for grad in grads]
        self.previous_cost = cost
        self.previous_weigths = [weigths.copy() for weigths in params]
        super().step(params, new_grads, cost)

    def reset(self):
        super().reset()
        self.previous_grads = None
        self.previous_cost = None
        self.previous_weigths = None
   
class Frac3Optimizer(ClassicOptimizer):
    
    def __init__(self, learning_rate=0.01, alpha_function = alpha_function, beta=0.9, verbose=False):
        super().__init__(learning_rate, verbose)
        self.alpha_func = alpha_function
        self.beta = 0.9
        self.previous_weigths = None
        self.previous_grads = None
        self.previous_cost = None
        
    def step(self, params, grads, cost):
        if self.previous_grads is None or self.previous_cost is None or self.previous_weigths is None:
            new_grads = grads
        else:
            norm_grad = 0
            for i in range(len(params)):
                norm_grad += np.linalg.norm(self.previous_grads[i]) ** 2
            norm_grad = np.sqrt(norm_grad)
            alpha = self.alpha_func(norm_grad, self.beta)
            new_grads = []
            for i in range(len(params)):
                new_grad = frac_gradient_from_gradient(alpha, self.previous_grads[i], params[i], self.previous_weigths[i])
                new_grads.append(new_grad)
            
        self.previous_grads = [ grad.copy() for grad in grads ]
        self.previous_cost = cost
        self.previous_weigths = [ weigths.copy() for weigths in params ]
        super().step(params, new_grads, cost)

    def reset(self):
        super().reset()
        self.previous_grads = None
        self.previous_cost = None
        self.previous_weigths = None

     
class FracAdap(FracOptimizer):
    def __init__(self, learning_rate=0.001, alpha_func=alpha_function, beta=0.9, increase_rate=1.1, decay_rate=0.6, verbose=False):
        super().__init__(learning_rate, alpha_func, beta, verbose)
        self.initial_learning_rate = learning_rate
        self.increase_rate = increase_rate
        self.decay_rate = decay_rate
        
    def step(self, params, grads, cost):
        self.compute_learning_rate(cost)
        super().step(params, grads, cost)
        
    def reset(self):
        super().reset()
        self.learning_rate = self.initial_learning_rate
            
    def compute_learning_rate(self, cost):
        if self.i == 0:
            self.learning_rate = self.initial_learning_rate
        elif cost < self.history['cost'][-1]:
            self.learning_rate *= self.increase_rate
        else:
            self.learning_rate *= self.decay_rate
        return self.learning_rate  
    
class FracTrue(ClassicOptimizer):
    
    def __init__(self, learning_rate=0.01, alpha_function = alpha_function, beta=0.9, verbose=False):
        super().__init__(learning_rate, verbose)
        self.alpha_func = alpha_function
        self.beta = 0.9
        self.previous_weigths = None
        self.previous_grads = None
        self.previous_cost = None
        self.parent = None
        
    def cost_function_alpha(self, alpha, params,grads):
        new_weights = []
        for i in range(len(params)):
            new_weights.append(params[i] - alpha * grads[i])
        if self.parent is None:
            raise ValueError("Parent NeuralNetwork is not set. Please set the parent before calling this method.")
        A_, _ = self.parent.forward_propagation(new_weights, self.parent.X) # type: ignore
        cost = self.parent.cost_function.cost(A_, new_weights, self.parent.y)
        return cost
            
    def step(self, params, grads, cost):
        if self.previous_grads is None or self.previous_cost is None or self.previous_weigths is None:
            new_grads = grads
        else:
            norm_grad = 0
            for i in range(len(params)):
                norm_grad += np.linalg.norm(self.previous_grads[i]) ** 2
            norm_grad = np.sqrt(norm_grad)
            alpha = self.alpha_func(norm_grad, self.beta)
            new_grads = []
            for i in range(len(params)):
                new_grad = frac_gradient_from_gradient(alpha, self.previous_grads[i], params[i], self.previous_weigths[i])
                new_grads.append(new_grad)
            
        self.learning_rate = minimize_scalar(
            self.cost_function_alpha, 
            args=(params, new_grads), 
            bounds=(0, 1), 
            method='bounded'
        ).x
        self.previous_grads = [ grad.copy() for grad in grads ]
        self.previous_cost = cost
        self.previous_weigths = [ weigths.copy() for weigths in params ]
        super().step(params, new_grads, cost)
        
    def reset(self):
        super().reset()
        self.previous_grads = None
        self.previous_cost = None
        self.previous_weigths = None