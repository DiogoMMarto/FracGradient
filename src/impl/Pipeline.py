from collections import defaultdict
from impl.NN import NeuralNetwork
from sklearn.metrics import classification_report , confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.use('Agg') 
plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'bold',
    'axes.labelweight': 'bold', # Specifically ensures axis labels are bold
    'axes.titleweight': 'bold'  # Specifically ensures titles are bold
})
plt.rcParams['figure.figsize'] = (12, 8)
import numpy as np
import os
import json

from itertools import product
from .Optimizers import FracOptimizerPsi

def get_scores(file_path):
    """Read classification report from a file and return it as a dictionary."""
    with open(file_path, 'r') as f:
        report = f.read()
    lines = report.split('\n')
    scores = {}
    for line in lines:
        if 'accuracy' in line:
            accuracy = float(line.split()[-2])
            scores['accuracy'] = accuracy
        if "macro avg" in line:
            parts = line.split()
            scores['macro avg'] = {
                'precision': float(parts[2]),
                'recall': float(parts[3]),
                'f1-score': float(parts[4]),
                'support': int(parts[5])
            }
    return scores

def extract_name(output: str):
        #split by "learning" and take the first part
        if "\\psi" in output:
            # split by word "$\\psi=$" and take the last part
            end = output.split("psi$=")[-1]
            if "sinh" in end:
                end = "sinh"
            if "sigmoid" in end:
                end = "sigmoid"
            if "x^" in end:
                end = "power"
            if "x e^" in end:
                end = "xex"
            if " x" in end:
                end = "ax"
            if "logexp" in end:
                end = "logexp"
            start = output.split("$\\theta$")[0].strip()
            return start + " " + end
        if "$\\theta$" in output:
            return output.split("$\\theta$")[0].strip()
        return output  

def end_pipeline_graphs(D, BASE_DIR,number_of_models_params,dataset_name,expirement_name):
    betas = []
    costs = []
    accs = []
    accs_test = []
    optimzer_names = []
    
    for Optimizer, params, output, name in D:
        params_path = output + "params.json"
        if not os.path.exists(params_path):
            with open(params_path, 'w') as f:
                json.dump(params, f, default=str)
    
    for Optimizer, params, output, name in D:
    # for dir in os.listdir(BASE_DIR):
        dir = output.replace(BASE_DIR, "")
        params_path = os.path.join(BASE_DIR, dir, "params.json")
        if not os.path.exists(params_path):
            print(f"Params file {params_path} does not exist, skipping.")
            continue
        # print(params_path)
        params = json.load(open(params_path))
        if "beta" in params:
            if not os.path.exists(os.path.join(BASE_DIR, dir, "history.json")):
                print(f"History file {os.path.join(BASE_DIR, dir, 'history.json')} does not exist, skipping.")
                continue
            history = json.load(open(os.path.join(BASE_DIR, dir, "history.json")))
            final_cost = history["cost"][-1]
            betas.append(params["beta"])
            costs.append(final_cost)
            classification_report = get_scores(os.path.join(BASE_DIR, dir, "classification_report.txt"))
            accs.append(classification_report["accuracy"])
            test_classification_report = get_scores(os.path.join(BASE_DIR, dir, "test_classification_report.txt"))
            accs_test.append(test_classification_report["accuracy"])
            optimzer_names.append(extract_name(name))
            
    # if the optimizers have params beta, plot the final cost vs beta
    plot_dir = os.path.join(BASE_DIR, "final_cost_vs_beta")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    unique_names = list(set(optimzer_names))
    colors = ["r", "b", "g", "c", "m", "y", "k", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    markers = ["x", "o", "*", "+", "v", "^", "s", "D", "<", ">", "p", "h"]

    # Ensure we don't run out of colors/markers if there are many optimizers
    name_to_color = {name: colors[i % len(colors)] for i, name in enumerate(unique_names)}
    name_to_marker = {name: markers[i % len(markers)] for i, name in enumerate(unique_names)}

    filtered_costs = [i for i in costs if i > 0] if len(costs) > 0 else [0]
    y_min = min(filtered_costs) - 0.1
    y_max = min(filtered_costs) + 1.5

    for name in unique_names:
        plt.figure()
        plt.xlabel("$\\beta$")
        plt.xscale("log")
        plt.ylabel("Loss")
        plt.title(f"Final Cost vs $\\beta$ - {name}")
        x_data = [betas[i] for i, n in enumerate(optimzer_names) if n == name]
        y_data = [costs[i] for i, n in enumerate(optimzer_names) if n == name]
        plt.scatter(
            x_data,
            y_data,
            c=name_to_color[name],
            marker=name_to_marker[name],
            label=name 
        )
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(plot_dir, f"{name}.png")
        plt.savefig(save_path)
        plt.close()
    
    plot_dir = os.path.join(BASE_DIR, "acc_vs_beta")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for name in unique_names:
        plt.figure()
        plt.xlabel("$\\beta$")
        plt.xscale("log")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs $\\beta$ - {name}")
        x_data = [betas[i] for i, n in enumerate(optimzer_names) if n == name]
        y_data = [accs[i] for i, n in enumerate(optimzer_names) if n == name]
        plt.scatter(
            x_data,
            y_data,
            c=name_to_color[name],
            marker=name_to_marker[name],
            label=name 
        )
        plt.ylim(min(accs) - 0.1, 1.05)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(plot_dir, f"{name}.png")
        plt.savefig(save_path)
        plt.close()
    
    plot_dir = os.path.join(BASE_DIR, "test_acc_vs_beta")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    m = min(accs_test) if len(accs_test) > 0 else 0
    for name in unique_names:
        plt.figure()
        plt.xlabel("$\\beta$")
        plt.xscale("log")
        plt.ylabel("Test Accuracy")
        plt.title(f"Test Accuracy vs $\\beta$ - {name}")
        x_data = [betas[i] for i, n in enumerate(optimzer_names) if n == name]
        y_data = [accs_test[i] for i, n in enumerate(optimzer_names) if n == name]
        plt.scatter(
            x_data,
            y_data,
            c=name_to_color[name],
            marker=name_to_marker[name],
            label=name 
        )
        plt.ylim(m - 0.05, 1.05)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(plot_dir, f"{name}.png")
        plt.savefig(save_path)
        plt.close()
        
    
    plot_dir = os.path.join(BASE_DIR, "beta_gamma_cost")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    groups = defaultdict(list)
    for Optimizer, params, output, name in D:
        if Optimizer == FracOptimizerPsi:
            history_path = os.path.join(output, "history.json")
            if not os.path.exists(history_path):
                continue
            with open(history_path) as f:
                history = json.load(f)
                
            final_cost = history["cost"][-1]
            pp = dict(params)
            beta = pp.get("beta", 1)
            n = pp.get("psi", "")
            n2 = n.n 
            name2 = str(n).split("_")[0]
            groups[name2].append((beta, n2, final_cost))

    for name2, points in groups.items():
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        betas, ns, costs = zip(*points)
        ax.scatter(betas, ns, costs, label=name2, s=50)
        ax.set_xlabel("$\\beta$")
        ax.set_ylabel("$\\gamma$")
        ax.set_zlabel("Loss")
        ax.set_title(f"3D Loss Surface: {name2}")
        ax.set_zlim(0, 2.5)
        ax.view_init(elev=10, azim=150) 
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(plot_dir, f"{name2}.png")
        plt.savefig(save_path)
        plt.close(fig)
        # plt.show()
        
        fig, ax = plt.subplots()
        sc = None
        for i, (name2, points) in enumerate(groups.items()):
            betas, ns, costs = zip(*points)
            sc = ax.scatter(ns, costs, c=betas, cmap='viridis', 
                            vmin=0, vmax=1,s=120, edgecolors='black', 
                            label=name2)

        cbar = plt.colorbar(sc) if sc else plt.colorbar()
        cbar.set_label('$\\beta$', fontweight='bold')

        ax.set_xlabel("$\\gamma$")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        ax.legend(loc='upper right')

        plt.title("Optimization Results (Color = Cost)")
        plt.savefig(BASE_DIR + "beta_n_cost_2d.png", bbox_inches='tight')
    
    plot_dir = os.path.join(BASE_DIR, "acc_vs_gamma")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    for name2, points in groups.items():
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        betas, ns, costs = zip(*points)
        accs = []
        for Optimizer, params, output, name in D:
            if Optimizer == FracOptimizerPsi:
                pp = dict(params)
                n = pp.get("psi", "")
                n2 = n.n 
                name3 = str(n).split("_")[0]
                if name3 == name2:
                    classification_report = get_scores(os.path.join(output, "classification_report.txt"))
                    accs.append(classification_report["accuracy"])
        ax.scatter(ns, accs, label=name2, s=50)
        ax.set_xlabel("$\\gamma$")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs $\\gamma$: {name2}")
        ax.set_ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(plot_dir, f"{name2}.png")
        plt.savefig(save_path)
        plt.close(fig)
        
    plot_dir = os.path.join(BASE_DIR, "final_costs_gamma")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    for name2, points in groups.items():
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        betas, ns, costs = zip(*points)
        ax.scatter(ns, costs, label=name2, s=50)
        ax.set_xlabel("$\\gamma$")
        ax.set_ylabel("Loss")
        ax.set_title(f"Final Costs vs $\\gamma$: {name2}")
        # ax.set_ylim(0, 2.5)
        ax.set_yscale("log")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(plot_dir, f"{name2}.png")
        plt.savefig(save_path)
        plt.close(fig)
        
    plot_dir = os.path.join(BASE_DIR, "test_acc_vs_gamma")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    for name2, points in groups.items():
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        betas, ns, costs = zip(*points)
        accs_test = []
        for Optimizer, params, output, name in D:
            if Optimizer == FracOptimizerPsi:
                pp = dict(params)
                n = pp.get("psi", "")
                n2 = n.n 
                name3 = str(n).split("_")[0]
                if name3 == name2:
                    test_classification_report = get_scores(os.path.join(output, "test_classification_report.txt"))
                    accs_test.append(test_classification_report["accuracy"])
        ax.scatter(ns, accs_test, label=name2, s=50)
        ax.set_xlabel("$\\gamma$")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Test Accuracy vs $\\gamma$: {name2}")
        ax.set_ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(plot_dir, f"{name2}.png")
        plt.savefig(save_path)
        plt.close(fig)
    
    def load_last_cost(output):
        if not os.path.exists(output + "history.json"):
            print(f"History file {output + 'history.json'} does not exist, skipping.")
            return float('inf')
        history = json.load(open(output + "history.json"))
        # scores = get_scores(output + "classification_report.txt")
        # if "accuracy" in scores:
        #     return -scores["accuracy"]
        # return 0
        last_cost = history["cost"][-1]
        # check if nan
        if np.isnan(last_cost):
            return float('inf')
        return last_cost
    
    last_cost = { (Optimizer, tuple(params.items()), output, name): load_last_cost(output) for Optimizer, params, output, name in D }
    sorted_last_cost = last_cost.items() #  sorted(last_cost.items(), key=lambda x: x[1])
    
    def extract_optimizer_class_name(Optimizer,params):
        pd = dict(params)
        if "psi" in pd:
            type_ = str(pd["psi"]).split("_")[0]
            return f"FracOptimizerPsi_{type_}"
        return Optimizer.__name__
    
    best_per_optimizer = {}
    for (Optimizer, params, output, name), cost in sorted_last_cost:
        optimizer_class_name = extract_optimizer_class_name(Optimizer, params)
        test_classification_report = get_scores(output + "test_classification_report.txt") if os.path.exists(output + "test_classification_report.txt") else None
        test_acc = test_classification_report["accuracy"] if test_classification_report else 0
        
        if optimizer_class_name not in best_per_optimizer:
            best_per_optimizer[optimizer_class_name] = (Optimizer, params, output, name, cost)
        else:
            existing_output = best_per_optimizer[optimizer_class_name][2]
            existing_test_report = get_scores(existing_output + "test_classification_report.txt") if os.path.exists(existing_output + "test_classification_report.txt") else None
            existing_test_acc = existing_test_report["accuracy"] if existing_test_report else 0
            
            if test_acc > existing_test_acc:
                best_per_optimizer[optimizer_class_name] = (Optimizer, params, output, name, cost)
                
    print("Best for each Optimizer class:")
    for Optimizer_name, (Optimizer, params, output, name, cost) in best_per_optimizer.items():
        print("-"*30)
        history = json.load(open(output + "history.json"))
        params = {k: v for k, v in params}
        classification_report = get_scores(output + "classification_report.txt")
        test_classification_report = get_scores(output + "test_classification_report.txt") if os.path.exists(output + "test_classification_report.txt") else {}
        print(f"{Optimizer_name}: {name} with cost {cost:.4f} and test accuracy {test_classification_report['accuracy']:.4f} at {output}")
        print(f"Parameters: {params}")
        try:
            with open("results/res.json", "r") as f:
                res = json.load(f)
                res = {} if res is None else res
                res[expirement_name] = res.get(expirement_name, {})
                res[expirement_name][Optimizer_name] = {
                    "name": name,
                    "params": params,
                    "last_cost": cost,
                    "dataset": dataset_name,
                    "number_of_models_params": number_of_models_params,
                    "expirement_name": expirement_name,
                    "output_dir": output,
                    "optimizer": Optimizer_name,
                    "time_to_train": history["time"][-1],
                    "classification_report": classification_report,
                    "test_classification_report": test_classification_report
                }
            with open("results/res.json", "w") as f:
                json.dump(res, f, indent=4 , default=str)
        except Exception as e:
            print(f"Error saving results to res.json: {e}")
            
    # extract only the best optimizers to D
    D = [ (Optimizer, params, output, name) for Optimizer, params, output, name, cost in best_per_optimizer.values() ]        
    
    plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    # plt.title("Cost function over Iterations - " + str(expirement_name))
    plt.tight_layout()
    y_heigth = float('inf')
    m = float('inf')
    S = 50
    
    for Optimizer , _ , output,name in D:
        history = json.load(open(output + "history.json"))
        # name = output.split("/")[-2]
        plt.plot(history["cost"], label=name)
        if history["cost"][S] < y_heigth:
            y_heigth = history["cost"][S]
        if history["cost"][-1] < m:
            m = history["cost"][-1]
    plt.ylim(ymin=m-0.1, ymax=y_heigth+0.1)
    plt.legend(loc='upper right')
    plt.savefig(BASE_DIR + "history.png")
    
    # similar plot but include x = time and y = cost
    plt.figure()
    plt.xlabel("Time")
    plt.ylabel("Loss")
    # plt.title("Cost function")
    plt.tight_layout()
    S = 250
    for Optimizer , _ , output,name in D:
        history = json.load(open(output + "history.json"))
        # name = output.split("/")[-2]
        plt.plot(history["time"], history["cost"], label=name)
    plt.ylim(ymin=m-0.1, ymax=y_heigth+0.1)
    plt.legend()
    plt.savefig(BASE_DIR + "history_time.png")    
      
      
def psi_graphs(D, BASE_DIR):
    
    vals = []
    
    for Optimizer, params, output, name in D:
        dir = output.replace(BASE_DIR, "")
        params_path = os.path.join(BASE_DIR, dir, "params.json")
        if not os.path.exists(params_path):
            print(f"Params file {params_path} does not exist, skipping.")
            continue
        params = json.load(open(params_path))
        if "beta" in params and "psi" in params:
            if not os.path.exists(os.path.join(BASE_DIR, dir, "history.json")):
                print(f"History file {os.path.join(BASE_DIR, dir, 'history.json')} does not exist, skipping.")
                continue
            history = json.load(open(os.path.join(BASE_DIR, dir, "history.json")))
            classification_report = get_scores(os.path.join(BASE_DIR, dir, "classification_report.txt"))
            test_classification_report = get_scores(os.path.join(BASE_DIR, dir, "test_classification_report.txt"))
            pp = dict(params)
            n = pp.get("psi")
            n = float(n.split("_")[-2])
            vals.append({
                "final_cost": history["cost"][-1],
                "beta": params["beta"],
                "acc": classification_report["accuracy"],
                "test_acc": test_classification_report["accuracy"],
                "gamma": n,
                "name": extract_name(name),
                "Optimizer": Optimizer,
            })
            
    df = pd.DataFrame(vals)
     
    plot_configs = [
        ("final_cost", "beta", "Loss", "final_cost_vs_beta.png", False),
        ("acc", "beta", "Accuracy", "acc_vs_beta.png", True),
        ("test_acc", "beta", "Test Accuracy", "test_acc_vs_beta.png", True),
        ("final_cost", "gamma", "Loss", "final_cost_vs_gamma.png", False),
        ("acc", "gamma", "Accuracy", "acc_vs_gamma.png", True),
        ("test_acc", "gamma", "Test Accuracy", "test_acc_vs_gamma.png", True)
    ]
    map_col = {
        "gamma": "$\\gamma$",
        "beta": "$\\beta$"
    }
     
    for y_col, x_col, y_label, filename, best_is_max in plot_configs:
        plt.figure()
        
        df_sorted = df.sort_values(y_col, ascending=not best_is_max)
        df_plot = df_sorted.drop_duplicates(["name", x_col])
        
        for label, group in df_plot.groupby("name"):
            group = group.sort_values(x_col)
            plt.scatter(group[x_col], group[y_col], marker="o", label=label)

        plt.xlabel(map_col.get(x_col,x_col))
        plt.ylabel(y_label)
        plt.legend()
        plt.tight_layout()
        if x_col == "gamma":
            plt.xlim(0,3.1)
        
        # Save to the BASE_DIR
        save_path = os.path.join(BASE_DIR, filename)
        plt.savefig(save_path)
        plt.close() # Close plot to free up memory
        print(f"Saved: {save_path}")
        print(df_plot)
    
    
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
        name_str = ""
        for k, v in params.items():
            if k == "learning_rate":
                k = "$\\theta$"
            if k == "beta":
                k = "$\\beta$"
            if k == "psi":
                k = "$\\psi$"
                v = v.__repr__()
                name_str += f" {k}={v}"
                continue
            name_str += f" {k}={v:.3f}"
        new_name = f"{name}{name_str}"
        
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
        print(f"Running pipeline, output dir: {self.output_dir}")        
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
        # plt.title('Confusion Matrix')
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
        plt.ylabel("Loss")
        # plt.title("Cost function")
        plt.savefig(history_cost_path)
        plt.close()
        
        plt.plot(history_time)
        plt.xlabel("Iteration")
        plt.ylabel("Time (s)")
        # plt.title("Time")
        plt.savefig(history_time_path)
        plt.close()
        
        # plot cost over time ( x = time , y = cost)
        plt.plot(history_time, history_cost)
        plt.xlabel("Time (s)")
        plt.ylabel("Loss")
        # plt.title("Cost function")
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
                axs[i].set_title(f'$\\alpha$ values for layer {i}')
                axs[i].set_xlabel('Iteration')
                axs[i].set_ylabel('$\\alpha$')
                axs[i].set_ylim(0, 1.02)
            # plt.suptitle('$\\alpha$ values for each layer over iterations')
            plt.xticks(fontsize=14, fontweight='bold')
            plt.yticks(fontsize=14, fontweight='bold')
            # plt.tight_layout()
            plt.savefig(self.output_dir + 'alpha_values.png')
            plt.close()
            
            # now plot the alpha values for each layer in a single plot
            plt.figure()
            for i in range(num_layers):
                plt.plot(history['alpha'][i], label=f'Layer {i}')
            # plt.title('$\\alpha$ values for each layer over iterations')
            plt.xlabel('Iteration')
            plt.ylabel('$\\alpha$')
            plt.ylim(0, 1.02)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir + 'alpha_values_all_layers.png')
            plt.close()
