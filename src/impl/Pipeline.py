from impl.NN import NeuralNetwork
from sklearn.metrics import classification_report , confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import numpy as np
import os
import json

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
        
    def run(self,epochs=100,verbose=False):
        if os.path.exists(self.output_dir):
            print("Output directory already exists. If you want to overwrite it, delete it first.")
            return
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
    