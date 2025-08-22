import os
from tensorflow.keras import models
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use Agg backend for matplotlib to avoid GUI issues in headless environments

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

class GradNormCollectorCallback(tf.keras.callbacks.Callback):
    """Callback that extracts grad norms from XLA-compatible optimizer."""
    
    def __init__(self):
        super().__init__()
        self.final_grad_norms = {}
    
    def on_train_end(self, logs=None):
        """Extract grad norms at the end of training."""
        optimizer = self.model.optimizer
        if hasattr(optimizer, 'get_grad_norms_history'):
            self.final_grad_norms = optimizer.get_grad_norms_history()
    
    def get_history(self):
        """Return the collected grad norms history."""
        return self.final_grad_norms
    
    
class LossPerIterationCallback(tf.keras.callbacks.Callback):
    """Callback to track loss per iteration."""
    
    def __init__(self):
        super().__init__()
        self.loss_history = []
    def on_train_batch_end(self, batch, logs=None):
        """Store loss at the end of each training batch."""
        if 'loss' in logs:
            self.loss_history.append(logs['loss'])
    def get_history(self):
        """Return the collected loss history."""
        return self.loss_history
        
# ---- Visualization Functions ----
def moving_average(values, window=1):
    if window <= 1:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")

def plot_gradient_norms(history, 
                        smooth=1, log_scale=True, 
                        per_layer=False, figsize=(10, 6)):
    """
    Plot gradient L2 norms tracked during training.

    Args:
        history (dict): {layer_name: [grad_norms]}
        smooth (int): Moving average window (default=1, no smoothing).
        log_scale (bool): Logarithmic y-axis for gradients.
        per_layer (bool): Whether to create separate subplots per layer.
        figsize (tuple): Figure size.
    """
    if not per_layer:
        plt.figure(figsize=figsize)
        for layer, norms in history.items():
            smoothed = moving_average(norms, smooth)
            plt.plot(smoothed, label=f"{layer} grad norm")
        plt.xlabel("Training Step")
        plt.ylabel("Gradient L2 Norm")
        if log_scale:
            plt.yscale("log")
        plt.title("Gradient Norm History")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # plt.show()
    else:
        n_layers = len(history)
        fig, axes = plt.subplots(n_layers, 1, figsize=(figsize[0], figsize[1]*n_layers))
        if n_layers == 1:
            axes = [axes]
        for ax, (layer, norms) in zip(axes, history.items()):
            smoothed = moving_average(norms, smooth)
            ax.plot(smoothed, label=f"{layer} grad norm")
            if log_scale:
                ax.set_yscale("log")
            ax.set_ylabel("Grad Norm")
            ax.legend()
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Training Step")
        plt.suptitle("Gradient Norms per Layer")
        plt.tight_layout()
        # plt.show()
        
def plot_loss_per_iteration(history,
                            smooth=1, log_scale=True,
                            figsize=(10, 6)):
    """
    Plot loss per iteration tracked during training.
    Args:
        history (list): List of loss values per iteration.
        smooth (int): Moving average window (default=1, no smoothing).
        log_scale (bool): Logarithmic y-axis for loss.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    smoothed = moving_average(history, smooth)
    plt.plot(smoothed, label="Loss per Iteration")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    if log_scale:
        plt.yscale("log")
    plt.title("Loss per Iteration History")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

from sklearn.metrics import confusion_matrix, classification_report

class TimePerEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = tf.timestamp()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = tf.timestamp() - self.epoch_start_time
        self.times.append(float(epoch_time))
        
class Pipeline:
    """
    Pipeline class for managing the training, evaluation, and saving/loading of a machine learning model.
    Attributes:
        X (np.ndarray): Training data features.
        y (np.ndarray): Training data labels.
        model (models.Model): The machine learning model to be trained and evaluated.
        name (str): Name of the pipeline. Defaults to "default_pipeline" if not provided.
        compile_kwargs (dict): Keyword arguments for compiling the model. Defaults to an empty dictionary.
        output_dir (str): Directory where outputs (e.g., model, history, evaluation results) will be saved. Defaults to "output".
        X_test (np.ndarray): Test data features. Optional.
        y_test (np.ndarray): Test data labels. Optional.
        data_augmentation (bool): Whether to apply data augmentation during training. Defaults to False.
        overwrite (bool): Whether to overwrite existing output directory. Defaults to False.
        continue_training (bool): Whether to continue training from a previously saved model. Defaults to False.
        history (dict): Training history, including metrics and loss over epochs.
        batch_size (int): Size of the batches used during training. Defaults to 32.
    Methods:
        save(): Saves the trained model and training history to the specified output directory.
        load(): Loads the model and training history from the specified output directory.
        evaluate(): Evaluates the model's predictions against the true labels, generating classification reports and confusion matrices. Saves these results to the specified output directory.
        run(epochs: int, verbose: bool): Trains the model on the provided training data for a specified number of epochs.
    """
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 model: models.Model,
                 name: str | None = None,
                 compile_kwargs: dict | None = None,
                 output_dir: str = "output",
                 X_test: np.ndarray | None = None,
                 y_test: np.ndarray | None = None,
                 data_augmentation: bool = False,
                 overwrite: bool = False,
                 continue_training: bool = False,
                 batch_size: int = 32,
                 dataset_name: str = "CIFAR-10",
                 expirement_name: str = "default_experiment"
                ):
        self.dataset = dataset_name 
        self.X = X
        self.y = y
        self.model = model
        self.name = name if name is not None else "default_pipeline"
        self.output_dir = output_dir
        self.X_test = X_test
        self.y_test = y_test
        self.data_augmentation = data_augmentation
        self.history = None
        self.overwrite = overwrite
        self.continue_training = continue_training
        self.compile_kwargs = compile_kwargs if compile_kwargs is not None else {}
        # if self.compile_kwargs.get("optimizer"):
        #     base_optimizer = self.compile_kwargs["optimizer"]
        #     track_grad = GradientNormTrackingOptimizer(base_optimizer)
        #     self.compile_kwargs["optimizer"] = track_grad
        # else:
        #     self.compile_kwargs["optimizer"] = GradientNormTrackingOptimizer(tf.keras.optimizers.Adam())
        self.batch_size = batch_size
        self.expirement_name = expirement_name
        
    def save(self):
        """
        Saves the model and training history to the specified output directory.
        """
        self.model.save(self.output_dir + '/model.h5')
        if self.history is not None:
            with open(self.output_dir + '/history.json', 'w') as f:
                json.dump(self.history, f)

    def load(self):
        """
        Loads the model and training history from the specified output directory.
        """
        self.model = models.load_model(self.output_dir + '/model.h5')
        with open(self.output_dir + '/history.json', 'r') as f:
            self.history = json.load(f)
        
    def evaluate(self):    
        """
        Evaluates the model's predictions against the true labels, generating classification reports and confusion matrices.
        Saves these results to the specified output directory.
        """
        y_pred = self.model.predict(self.X)
        y_true = np.argmax(self.y, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)

        report = classification_report(y_true, y_pred_classes)
        cm = confusion_matrix(y_true, y_pred_classes)

        os.makedirs(self.output_dir, exist_ok=True)

        with open(self.output_dir + '/classification_report.txt', 'w') as f:
            f.write(str(report))

        with open(self.output_dir + '/confusion_matrix.json', 'w') as f:
            json.dump(cm.tolist(), f)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest')
        plt.title(f'Confusion Matrix - {self.name}')
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(y_true)))
        plt.xticks(tick_marks, np.unique(y_true), rotation=45)
        plt.yticks(tick_marks, np.unique(y_true))
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        plt.savefig(self.output_dir + '/confusion_matrix.png')
        plt.close()

        if self.X_test is not None and self.y_test is not None:
            y_test_pred = self.model.predict(self.X_test)
            y_test_true = np.argmax(self.y_test, axis=1)
            y_test_pred_classes = np.argmax(y_test_pred, axis=1)
            test_report = classification_report(y_test_true, y_test_pred_classes)

            with open(self.output_dir + '/test_classification_report.txt', 'w') as f:
                f.write(str(test_report))
                
            test_cm = confusion_matrix(y_test_true, y_test_pred_classes)
            with open(self.output_dir + '/test_confusion_matrix.json', 'w') as f:
                json.dump(test_cm.tolist(), f)
                
            plt.figure(figsize=(10, 8))
            plt.imshow(test_cm, interpolation='nearest')
            plt.title(f'Test Confusion Matrix - {self.name}')
            plt.colorbar()
            tick_marks = np.arange(len(np.unique(y_test_true)))
            plt.xticks(tick_marks, np.unique(y_test_true), rotation=45)
            plt.yticks(tick_marks, np.unique(y_test_true))
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.tight_layout()
            plt.savefig(self.output_dir + '/test_confusion_matrix.png')
            plt.close()
            
        if self.history is not None:
            history_cost = self.history.get('loss', [])
            history_accuracy = self.history.get('accuracy', [])
            history_time = self.history.get('time', [])
            history_cost_validation = self.history.get('val_loss', [])
            history_accuracy_validation = self.history.get('val_accuracy', [])
            history_grad_norms = self.history.get('grad_norms', {})
            history_loss_per_iteration = self.history.get('loss_per_iteration', [])
            
            if history_loss_per_iteration:
                plot_loss_per_iteration(history_loss_per_iteration,
                                        smooth=1, 
                                        log_scale=False,
                                        figsize=(10, 6))
                plt.savefig(self.output_dir + '/loss_per_iteration.png')
                plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(history_cost, label='Training Loss')
            plt.title('Training Loss Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(self.output_dir + '/training_loss.png')
            plt.close()

            if history_accuracy:
                plt.figure(figsize=(10, 5))
                plt.plot(history_accuracy, label='Training Accuracy')
                plt.title('Training Accuracy Over Epochs')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.savefig(self.output_dir + '/training_accuracy.png')
                plt.close()

            if history_time:
                plt.figure(figsize=(10, 5))
                plt.plot(history_time, label='Time per Epoch')
                plt.title('Time per Epoch Over Training')
                plt.xlabel('Epochs')
                plt.ylabel('Time (seconds)')
                plt.legend()
                plt.savefig(self.output_dir + '/time_per_epoch.png')
                plt.close()
        
            if history_time and history_cost:
                plt.figure(figsize=(10, 5))
                plt.plot(history_time, history_cost)
                plt.title('Cost Function Over Time')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Cost')
                plt.savefig(self.output_dir + '/cost_function_over_time.png')
                plt.close()
                
            if history_cost_validation:
                plt.figure(figsize=(10, 5))
                plt.plot(history_cost_validation, label='Validation Loss')
                plt.title('Validation Loss Over Epochs')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(self.output_dir + '/validation_loss.png')
                plt.close()
                
            if history_accuracy_validation:
                plt.figure(figsize=(10, 5))
                plt.plot(history_accuracy_validation, label='Validation Accuracy')
                plt.title('Validation Accuracy Over Epochs')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.savefig(self.output_dir + '/validation_accuracy.png')
                plt.close()
                
            if len(history_grad_norms) > 0:
                plot_gradient_norms(history_grad_norms, 
                                    smooth=1, 
                                    log_scale=False, 
                                    per_layer=False, 
                                    figsize=(10, 6))
                plt.savefig(self.output_dir + '/gradient_norms.png')
                plt.close()
                
                plot_gradient_norms(history_grad_norms,
                                    smooth=1,
                                    log_scale=False,
                                    per_layer=True,
                                    figsize=(10, 6))
                plt.savefig(self.output_dir + '/gradient_norms_per_layer.png')
                plt.close()
                
        params = self.compile_kwargs.get("optimizer", {}).get_config() if self.compile_kwargs.get("optimizer") else {}
        Optimizer_name = params.get("name", "Unknown Optimizer")
        number_of_models_params = sum(np.prod(v.shape) for v in self.model.trainable_variables)
        cost = history_cost[-1] if history_cost else None
        print(f"Optimizer: {Optimizer_name}, Number of Parameters: {number_of_models_params}, Final Cost: {cost}, Params: {params}")
        dataset_name = self.dataset 
        if "beta" in params:
            with open("results/beta_results.txt", "a") as f:
                f.write(f"{Optimizer_name}    {params['beta']}    {number_of_models_params}    {cost}    {dataset_name}\n")

    def report_to_json(self):
        params = self.compile_kwargs.get("optimizer", {}).get_config() if self.compile_kwargs.get("optimizer") else {}
        Optimizer_name = params.get("name", "Unknown Optimizer")
        def load_history():
            if not os.path.exists(self.output_dir + "history.json"):
                return {}
            with open(self.output_dir + "history.json", "r") as f:
                return json.load(f)
        history = load_history()
        final_cost = history.get("loss", [None])[-1]
        time = history.get("time", [])
        # remove outliers from time
        n = len(time)
        mean_time = np.mean(time)
        std_time = np.std(time)
        time = [t for t in time if abs(t - mean_time) < 3 * std_time]
        new_mean_time = np.mean(time)
        final_time = new_mean_time * n
        
        classification_report = get_scores(self.output_dir + "classification_report.txt")
        test_classification_report = get_scores(self.output_dir + "test_classification_report.txt") 
        
        number_of_models_params = sum(np.prod(v.shape) for v in self.model.trainable_variables)
        
        with open("results/res.json", "r") as f:
            res = json.load(f)
            res = {} if res is None else res
            betav = params.get('beta', '')
            res[Optimizer_name + str(betav) + ","+ self.expirement_name] = {
                "name": self.name,
                "params": params,
                "last_cost": final_cost,
                "dataset": self.dataset,
                "number_of_models_params": int(number_of_models_params),
                "expirement_name": self.expirement_name,
                "optimizer": Optimizer_name,
                "output_dir": self.output_dir,
                "time_to_train": final_time,
                "classification_report": classification_report,
                "test_classification_report": test_classification_report
            }
        with open("results/res.json", "w") as f:
            json.dump(res, f, indent=4)
        
    def run(self, epochs=100, verbose=False):
        """
        Trains the model on the provided training data for a specified number of epochs.
        If the output directory already exists and `overwrite` is False, it will not proceed with training.
        """
        if not self.overwrite and tf.io.gfile.exists(self.output_dir + '/model.h5'):
            print("Output directory already exists. If you want to overwrite it, set `overwrite=True`.")
            self.report_to_json()
            return
        
        if self.continue_training:
            self.load()
        
        callbacks = [TimePerEpochCallback(),GradNormCollectorCallback(),LossPerIterationCallback()]
        self.model.compile(**self.compile_kwargs)
        
        if self.data_augmentation:
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.15,
                height_shift_range=0.15,
                horizontal_flip=True,
                zoom_range=0.1,
                channel_shift_range=0.12,
            )
            data_gen.fit(self.X)
            self.history = self.model.fit(
                data_gen.flow(self.X, self.y, batch_size=self.batch_size),
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_data=(self.X_test, self.y_test) if self.X_test is not None and self.y_test is not None else None,
                # steps_per_epoch=len(self.X) // self.batch_size
            )
        else:
            self.history = self.model.fit(
                self.X,
                self.y,
                batch_size=self.batch_size,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_data=(self.X_test, self.y_test) if self.X_test is not None and self.y_test is not None else None
            )
        self.history = self.history.history
        self.history['time'] = [callback.times for callback in callbacks if isinstance(callback, TimePerEpochCallback)][0]
        self.history['grad_norms'] = [callback.get_history() for callback in callbacks if isinstance(callback, GradNormCollectorCallback)][0]
        self.history['loss_per_iteration'] = [callback.get_history() for callback in callbacks if isinstance(callback, LossPerIterationCallback)][0]
        os.makedirs(self.output_dir, exist_ok=True)
        self.evaluate()
        self.save()
        print(f"Training completed and results saved to {self.output_dir}.")
        
        