import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
import numpy as np

@tf.function
def alpha_function(norm_GradCost, beta):
    """
    Computes the alpha value based on the norm of the gradient cost.
    """
    return 1.0 - (2.0 / np.pi) * tf.math.atan(norm_GradCost * beta)

@tf.keras.utils.register_keras_serializable(package="CustomOptimizers")
class FracOptimizer(Optimizer):
    def __init__(self, 
                 learning_rate=0.001, 
                 beta=0.9, 
                 alpha_func=None, # Changed to None default
                 name="FracOptimizer", 
                 **kwargs):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        
        self.beta = beta
        
        if isinstance(alpha_func, str):
            # Map string name back to the actual function
            # Ensure 'alpha_function' is available in the global scope
            if alpha_func == "alpha_function":
                self.alpha_func = alpha_function 
            else:
                raise ValueError(f"Unknown alpha_func: {alpha_func}")
        else:
            self.alpha_func = alpha_func or alpha_function

        self.grad_norms_current_step = {}
        self.grad_norm_storage = {}
        self.grad_norm_counters = {}
        self.max_storage_size = 150 * 1563
        self._built = False

    def build(self, var_list):
        if self._built:
            return
        super().build(var_list)
        
        self.prev_weights = []
        self.prev_grads = []

        for i, var in enumerate(var_list):
            print(f"  {i}: {var.name} -> index {i}")
            self.prev_weights.append(
                self.add_variable_from_reference(
                    reference_variable=var, 
                    name=f"prev_weight_{i}", 
                    initializer="zeros"
                )
            )
            self.prev_grads.append(
                self.add_variable_from_reference(
                    reference_variable=var, 
                    name=f"prev_grad_{i}",
                    initializer="zeros"
                )
            )
            self.grad_norm_storage[i] = self.add_variable(
                shape=(self.max_storage_size,),
                dtype=tf.float32,
                name=f"grad_norms_{i}",
                initializer='zeros'
            )
            self.grad_norm_counters[i] = self.add_variable(
                shape=(),
                dtype=tf.int32,
                name=f"grad_norm_counter_{i}",
                initializer='zeros'
            )
        self._built = True

    def _store_grad_norm(self, layer_name, alpha_value): # FOR TRACKING GRAD NORM
        """Store grad norm in TF variable (XLA compatible)."""
        counter = self.grad_norm_counters[layer_name]
        storage = self.grad_norm_storage[layer_name]
        
        # Use modulo to wrap around if we exceed storage size
        index = counter % self.max_storage_size
        # Update the storage
        storage[index].assign(alpha_value)
        
        # Increment counter
        self.grad_norm_counters[layer_name].assign_add(1)
        
        return storage
   
    def update_step(self, gradient, variable, learning_rate):
            """
            Applies a single optimization step on a given variable.
            Performs standard SGD on the first iteration, then fractional gradient.

            Args:
                gradient: The gradient of the loss with respect to the variable.
                variable: The variable to be updated.
                learning_rate: The current learning rate.
            """
            variable_index = self._get_variable_index(variable) 

            prev_weight = self.prev_weights[variable_index]
            prev_grad = self.prev_grads[variable_index]

            lr = tf.cast(learning_rate, variable.dtype) 

            is_first_iteration = tf.equal(self._iterations, 0)

            def standard_sgd_update():
                self.assign_sub(variable, lr * gradient)
                
            def fractional_gradient_update():
                norm_grad = tf.norm(prev_grad)
                alpha = self.alpha_func(norm_grad, self.beta)
                
                self._store_grad_norm(variable_index, alpha) #    FOR TRACKING GRAD NORM

                diff = tf.abs(variable - prev_weight)
        
                gamma_val = tf.math.exp(tf.math.lgamma(2.0 - alpha))
                frac_grad = prev_grad * tf.pow(diff + tf.keras.backend.epsilon(), 1.0 - alpha) / gamma_val

                self.assign_sub(variable, lr * frac_grad)


            # Use tf.cond to execute the appropriate update based on the iteration
            tf.cond(is_first_iteration, standard_sgd_update, fractional_gradient_update)
            self.assign(prev_weight, variable)
            self.assign(prev_grad, gradient)

    def get_grad_norms_history(self): # FOR TRACKING GRAD NORM
        """Extract grad norms history as numpy arrays."""
        history = {}
        for layer_name in self.grad_norm_storage:
            counter_val = int(self.grad_norm_counters[layer_name].numpy())
            stored_values = self.grad_norm_storage[layer_name].numpy()
            
            # Only take the values that were actually stored
            if counter_val <= self.max_storage_size:
                history[layer_name] = stored_values[:counter_val].tolist()
            else:
                # If we wrapped around, we need to reconstruct the order
                start_idx = counter_val % self.max_storage_size
                history[layer_name] = (
                    stored_values[start_idx:].tolist() + 
                    stored_values[:start_idx].tolist()
                )
        
        return history
    
    def get_config(self):
        # 3. Use super().get_config() and ensure values are serializable
        config = super().get_config()
        config.update({
            "beta": self.beta,
            "alpha_func": self.alpha_func.__name__ if hasattr(self.alpha_func, '__name__') else str(self.alpha_func)
        })
        return config