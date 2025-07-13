from impl.NN import NeuralNetwork
from sklearn.metrics import classification_report , confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import numpy as np
import os
import json

from itertools import product

def end_pipeline_graphs(D, BASE_DIR):
    plt.figure(figsize=(12, 8))
    plt.xlabel("Iteration")
    plt.ylabel("$J(\\Theta)$")
    plt.title("Cost function using Gradient Descent")
    plt.tight_layout()
    y_heigth = 100
    S = 10
    for Optimizer , _ , output,name in D:
        history = json.load(open(output + "history.json"))
        # name = output.split("/")[-2]
        plt.plot(history["cost"], label=name)
        if history["cost"][S] < y_heigth:
            y_heigth = history["cost"][S]
    plt.ylim(ymin=0, ymax=y_heigth)
    plt.legend()
    plt.savefig(BASE_DIR + "history.png")
    
    # similar plot but include x = time and y = cost
    plt.figure(figsize=(12, 8))
    plt.xlabel("Time")
    plt.ylabel("$J(\\Theta)$")
    plt.title("Cost function using Gradient Descent")
    plt.tight_layout()
    y_heigth = 0
    S = 10
    for Optimizer , _ , output,name in D:
        history = json.load(open(output + "history.json"))
        # name = output.split("/")[-2]
        plt.plot(history["time"], history["cost"], label=name)
        if history["cost"][S] > y_heigth:
            y_heigth = history["cost"][S]
    plt.ylim(ymin=0, ymax=y_heigth)
    plt.legend()
    plt.savefig(BASE_DIR + "history_time.png")
    
    min_cost = float('inf')
    best_optimizer = None
    for Optimizer, _, output, name in D:
        history = json.load(open(output + "history.json"))
        final_cost = history["cost"][-1]
        if final_cost < min_cost:
            min_cost = final_cost
            best_optimizer = name
    print(f"The best optimizer is {best_optimizer} with a final cost of {min_cost:.4f}")
    
    # if the optimizers have params beta, plot the final cost vs beta
    plt.figure(figsize=(12, 8))
    plt.xlabel("Beta")
    # logscale x-axis
    plt.xscale("log")
    plt.ylabel("$J(\\Theta)$")
    # limit y-axis to [0, 1]
    plt.ylim(0.2, 0.5)
    plt.title("Final Cost vs Beta")
    plt.tight_layout()
    betas = []
    costs = []
    for Optimizer, params, output, name in D:
        if "beta" in params:
            history = json.load(open(output + "history.json"))
            final_cost = history["cost"][-1]
            betas.append(params["beta"])
            costs.append(final_cost)
    plt.scatter(betas, costs)
    # plt.plot(betas, costs, label="Final Cost vs Beta")
    plt.legend()
    plt.savefig(BASE_DIR + "final_cost_vs_beta.png")

def expand_tuple(t):
    elements = [item if isinstance(item, list) else [item] for item in t]
    return list(product(*elements))

from itertools import product

def expand_dict_combinations(d):
    # Separate keys with list values and non-list values
    keys = []
    values = []
    for k, v in d.items():
        if isinstance(v, list):
            keys.append(k)
            values.append(v)
        else:
            keys.append(k)
            values.append([v])  # wrap non-list values for consistent processing
    
    # Create all combinations (Cartesian product)
    combinations = product(*values)
    
    # Rebuild dictionaries from combinations
    return [dict(zip(keys, combo)) for combo in combinations]

def gen_names(d: list[tuple]):
    ret = []
    for i in range(len(d)):
        opt = d[i][0]
        params: dict = d[i][1]
        output_dir: str = d[i][2]
        name:str  = d[i][3]
        
        params_str = "_".join(f"{k}_{v}" for k, v in params.items())
        new_output_dir = output_dir[:-1] + f"{params_str}/"
        new_name = f"{name} {params_str.replace('_', ' ')}"
        
        new_tuple = (opt, params, new_output_dir, new_name)
        ret.append(new_tuple)
    return ret

def gen_grid_search(d: list[tuple]):
    expanded = [ (d[i][0], expand_dict_combinations(d[i][1]), d[i][2], d[i][3]) for i in range(len(d)) ]
    expanded = [ expand_tuple(t) for t in expanded ]
    expanded = [ item for sublist in expanded for item in sublist ]  # Flatten
    return gen_names(expanded)

class Pipeline:
    """A pipeline for training and evaluating a neural network model.
    This class handles the training of the model, evaluation of its performance, and saving of results such as classification reports, confusion matrices, and training history.
    It also supports optional testing on a separate test dataset if provided.
    
    Parameters
    ----------
    X : np.ndarray
        The input features for training the model, with shape (number of examples, number of features).
    y : np.ndarray
        The target labels for training the model, with shape (number of examples, number of classes).
    model : NeuralNetwork
        An instance of the NeuralNetwork class that defines the architecture and training parameters of the model.
    output_dir : str
        The directory where the results will be saved. If the directory already exists, the training will not proceed to avoid overwriting.
    X_test : np.ndarray | None, optional
        The input features for testing the model, with shape (number of examples, number of features). Default is None.
    y_test : np.ndarray | None, optional
        The target labels for testing the model, with shape (number of examples, number of classes). Default is None.
    
    Methods
    -------
    run(epochs=100, verbose=False)
        Trains the model on the provided training data for a specified number of epochs.
        If the output directory already exists, it will not proceed with training.
    evaluate(y_pred)
        Evaluates the model's predictions against the true labels, generating classification reports, confusion matrices, and training history plots.
        Saves these results to the specified output directory.
    
    Attributes
    ----------
    X : np.ndarray
        The input features for training the model.
    y : np.ndarray
        The target labels for training the model.
    model : NeuralNetwork
        The neural network model to be trained and evaluated.
    output_dir : str
        The directory where the results will be saved.
    X_test : np.ndarray | None
        The input features for testing the model, if provided.
    y_test : np.ndarray | None
        The target labels for testing the model, if provided.
    """
    def __init__(self, X: np.ndarray , y: np.ndarray , model: NeuralNetwork, output_dir: str, X_test: np.ndarray | None = None, y_test: np.ndarray | None = None):
        self.X = X
        self.y = y
        self.model = model
        self.output_dir = output_dir
        self.X_test = X_test	
        self.y_test = y_test
        
    def load_weigths_and_history(self):
        """Load weights and history from the output directory if they exist."""
        if not os.path.exists(self.output_dir):
            print(f"Output directory {self.output_dir} does not exist.")
            return
        
        # Load weights
        self.model.weights = []
        for i in range(len(self.model.layers)+1):
            weights_path = os.path.join(self.output_dir, f'weights_{i}.npy')
            if os.path.exists(weights_path):
                self.model.weights.append(np.load(weights_path))
            else:
                print(f"Weight file {weights_path} does not exist.")
        
        # Load history
        history_path = os.path.join(self.output_dir, 'history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.model.optimizer.history = json.load(f)
        else:
            print(f"History file {history_path} does not exist.")
        
    def run(self,epochs=100,verbose=False):
        if os.path.exists(self.output_dir):
            print("Output directory already exists. If you want to overwrite it, delete it first.")
            return
            self.load_weigths_and_history()
            print("Loaded existing weights and history.")
        else:
            self.model.fit(self.X, self.y, epochs=epochs, verbose=verbose)
        y_pred = self.model.predict(self.X)
        self.evaluate(y_pred)
        
    def evaluate(self,y_pred):
        classification_report_path = self.output_dir + 'classification_report.txt'
        y_true = np.argmax(self.y, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        report = classification_report(y_true, y_pred)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(classification_report_path, 'w') as f:
            f.write(str(report))
            
        if self.X_test is not None and self.y_test is not None:
            y_test_pred = self.model.predict(self.X_test)
            y_test_true = np.argmax(self.y_test, axis=1)
            y_test_pred = np.argmax(y_test_pred, axis=1)
            test_report = classification_report(y_test_true, y_test_pred)
            test_report_path = self.output_dir + 'test_classification_report.txt'
            with open(test_report_path, 'w') as f:
                f.write(str(test_report))
            
        cm = confusion_matrix(y_true, y_pred)
        cm_path = self.output_dir + 'confusion_matrix.png'
        # make it so the plot doesnt appear on screen
        plt.figure(figsize=(10, 10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.imshow(cm, interpolation='nearest')
        plt.colorbar()
        plt.savefig(cm_path)
        plt.close()
        
        history = self.model.optimizer.get_history()
        history_cost = history['cost']
        history_time = history['time']
        history_cost_path = self.output_dir + 'history_cost.png'
        history_time_path = self.output_dir + 'history_time.png'
        plt.plot(history_cost)
        plt.xlabel("Iteration")
        plt.ylabel("$J(\\Theta)$")
        plt.title("Cost function using Gradient Descent")
        plt.savefig(history_cost_path)
        plt.close()
        plt.plot(history_time)
        plt.xlabel("Iteration")
        plt.ylabel("Time (s)")
        plt.title("Time using Gradient Descent")
        plt.savefig(history_time_path)
        plt.close()
        
        # plot cost over time ( x = time , y = cost)
        plt.plot(history_time, history_cost)
        plt.xlabel("Time (s)")
        plt.ylabel("$J(\\Theta)$")
        plt.title("Cost function using Gradient Descent")
        plt.savefig(self.output_dir + 'cost_function.png')
        plt.close()
        
        history_path = self.output_dir + 'history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f)
        
        for i in range(len(self.model.weights)):
            weights_path = self.output_dir + f'weights_{i}.npy'
            np.save(weights_path, self.model.weights[i])
        
        print(f"Saved results to {self.output_dir}")
        
        if "alpha" in history:
            # create a plot with number of layers of subplots each plotting the alpha values for each layer
            num_layers = len(history['alpha'])
            fig, axs = plt.subplots(num_layers, 1, figsize=(12, 6 * num_layers))
            if num_layers == 1:
                axs = [axs]
            for i in range(num_layers):
                axs[i].plot(history['alpha'][i])
                axs[i].set_title(f'Alpha values for layer {i}')
                axs[i].set_xlabel('Iteration')
                axs[i].set_ylabel('Alpha')
                axs[i].set_ylim(0, 1.02)
            plt.suptitle('Alpha values for each layer over iterations')
            plt.tight_layout()
            plt.savefig(self.output_dir + 'alpha_values.png')
            plt.close()