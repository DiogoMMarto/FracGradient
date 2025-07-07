import os
from tensorflow.keras import models
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use Agg backend for matplotlib to avoid GUI issues in headless environments

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
                 batch_size: int = 32):
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
        self.batch_size = batch_size
        
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
  
    def run(self, epochs=100, verbose=False):
        """
        Trains the model on the provided training data for a specified number of epochs.
        If the output directory already exists and `overwrite` is False, it will not proceed with training.
        """
        if not self.overwrite and tf.io.gfile.exists(self.output_dir + '/model.h5'):
            print("Output directory already exists. If you want to overwrite it, set `overwrite=True`.")
            return
        
        if self.continue_training:
            self.load()
        
        callbacks = [TimePerEpochCallback()]
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
        os.makedirs(self.output_dir, exist_ok=True)
        self.evaluate()
        self.save()
        print(f"Training completed and results saved to {self.output_dir}.")
        
        