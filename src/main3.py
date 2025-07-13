from impl.Pipeline import Pipeline , gen_grid_search , end_pipeline_graphs
from impl.NN import NeuralNetwork
from impl.Optimizers import ClassicOptimizer , AdaptiveLearningRateOptimizer, FracTrue , MomentumOptimizer , FracOptimizer , FracOptimizer2 , AdamOptimizer , FracAdap , Frac3Optimizer, FracOptimizerBStable
from impl.CostFunctions import BinaryCrossEntropy , L2Regularization , ActivationFunction
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import json
from pathlib import Path
import h5py
from numpy import ndarray

DATASET_PATH = "datasets/Happy_datasets/datasets/"
BASE_DIR = "results/output_HappyFace_2/"
NUM_EPOCHS = 3000
VERBOSE = True

def one_hot(y):
    one_hot = np.zeros((y.shape[0], y.max() + 1))
    for i in range(y.shape[0]):
        one_hot[i][y[i]] = 1
    return one_hot

# def load_dataset():
#     # labels is the directory names in DATASET_PATH
#     labels = [d.name for d in os.scandir(DATASET_PATH) if d.is_dir()]
#     labels_map = {label: i for i, label in enumerate(labels)}
#     images = []
#     labels_list = []
#     # open each directory in DATASET_PATH and load the images png
#     for label in labels:
#         label_path = os.path.join(DATASET_PATH, label)
#         for img_path in glob.glob(os.path.join(label_path, "*.png")):
#             img = plt.imread(img_path)
#             images.append(img.flatten())
#             labels_list.append(int(labels_map[label]))
#     return np.array(images), np.array(labels_list) , labels_map
    
def load_dataset():
    # .h5 format on test_happy.h5 and train_happy.h5
    train_path = Path(DATASET_PATH) / "train_happy.h5"
    test_path = Path(DATASET_PATH) / "test_happy.h5"
    with h5py.File(train_path, 'r') as f:
        X_train: ndarray = f['train_set_x'][:] # type: ignore
        y_train: ndarray = f['train_set_y'][:]  # type: ignore
        list_classes: list = f['list_classes'][:]  # type: ignore
        
    with h5py.File(test_path, 'r') as f:
        X_test: ndarray = f['test_set_x'][:]  # type: ignore
        y_test: ndarray = f['test_set_y'][:]  # type: ignore
        list_classes_test: list = f['list_classes'][:]  # type: ignore
        
    # flatten X from (m, 64, 64, 3) to (m, 12288)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    # convert y from (m,1) to one-hot encoding
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)
    # check if the classes are the same
    if not np.array_equal(list_classes, list_classes_test):
        raise ValueError("The classes in the train and test sets are not the same.")

    return X_train , y_train, X_test, y_test, list_classes
    
def main():
    # Load dataset
    X, y, X_test, y_test, labels = load_dataset()

    p_gen = lambda Optimizer,params,output: Pipeline(
        X, 
        y, 
        NeuralNetwork(
            [64], 
            X.shape[1], 
            y.shape[1], 
            BinaryCrossEntropy(
                regularization=L2Regularization(0.1),
                activation_function_names=[
                    "sigmoid",
                    "sigmoid",
                ]
            ), 
            Optimizer(**params)
        ),
        output,
        X_test=X_test,
        y_test=y_test
    )
    
    D = [
        ( ClassicOptimizer, {"learning_rate":0.1}, BASE_DIR + "classical/" , "Gradient Descent"),
        ( AdaptiveLearningRateOptimizer, {"initial_learning_rate":1}, BASE_DIR + "adaptive/", "Adaptive Learning Rate"),
        # ( MomentumOptimizer, {"learning_rate":1, "momentum":0.5}, BASE_DIR + "momentum/"),
        # ( FracOptimizer, {"learning_rate":1}, BASE_DIR + "frac/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.1}, BASE_DIR + "fracB01/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.01}, BASE_DIR + "fracB001/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.001}, BASE_DIR + "fracB0001/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":10}, BASE_DIR + "fracB10/"),
        # ( FracOptimizer, {"learning_rate":1,"beta":0.5}, BASE_DIR + "fracB05/"),
        ( FracAdap, {"learning_rate":1,"beta":1}, BASE_DIR + "fracAdapB1/" , "FracGradient V2 Adaptive"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.5}, BASE_DIR + "frac3B05/"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.05}, BASE_DIR + "frac3B005/"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.1}, BASE_DIR + "frac3B01/"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":0.005}, BASE_DIR + "frac3B0005/"),
        # ( Frac3Optimizer, {"learning_rate":1,"beta":5}, BASE_DIR + "frac3B5/"),
        # ( Frac3Optimizer, {"learning_rate":0.5,"beta":50}, BASE_DIR + "frac3B50/", "Fractional Gradient Descent V3"),
        # ( FracOptimizer2, {"learning_rate":1}, BASE_DIR + "frac2/"),
        # ( FracOptimizer2, {"learning_rate":1,"beta":0}, BASE_DIR + "frac2B0/"),
        # ( FracOptimizer2, {"learning_rate":1,"beta":0.1}, BASE_DIR + "frac2B01/"),
        ( FracOptimizer, {"learning_rate":0.1,"beta":5}, BASE_DIR + "fracB5/" , "FracGradient V2"),
        # ( AdamOptimizer, {"learning_rate":1}, BASE_DIR + "adam/"),
        # ( FracTrue, {"beta":0.5,"verbose":True}, BASE_DIR + "fracTrue/"),
        ( FracOptimizer2, {"learning_rate":0.1,"beta":1}, BASE_DIR + "frac2B01/", "FracGradient"),
        ( FracOptimizerBStable, {"learning_rate":0.1,"beta":0.05}, BASE_DIR + "fracBStable001_/", "FracGradient B Stable"),
    ]
    
    D = gen_grid_search(
        [
         (FracOptimizer , {"learning_rate":[10,5,2,1,0.1,0.01,0.001],"beta":[5,1,0.5,0.1,0.05,0.01,0.005,0.001]}, BASE_DIR + "_frac_v2_/", "FracGradient V2"),
         (FracAdap , {"learning_rate":[5,2,1],"beta":[5,1,0.5,0.1,0.05,0.01]}, BASE_DIR + "_frac_adap_v2/", "FracGradient V2 Adaptive"),
         (FracOptimizer , {"learning_rate":[0.1],"beta":list(2**np.arange(-10,3,0.3))}, BASE_DIR + "_frac_v2_/", "FracGradient V2"),
         (FracAdap , {"learning_rate":[0.1],"beta":list(2**np.arange(-10,3,0.3))}, BASE_DIR + "_frac_adap_v2/", "FracGradient V2 Adaptive"),
        ]
    )
    
    def run_pipeline(Optimizer,params,output):
        p = p_gen(Optimizer,params,output)
        p.run(epochs=NUM_EPOCHS,verbose=VERBOSE)
   
    if False:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_pipeline, Optimizer,params,output) for Optimizer,params,output,_ in D]
            for future in futures:
                future.result()
    else:
        for Optimizer, params, output, _ in D:
            run_pipeline(Optimizer, params, output)
    
    end_pipeline_graphs(D, BASE_DIR)
    
if __name__ == "__main__":
    main()    
    