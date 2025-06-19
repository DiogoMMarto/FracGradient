from scipy.io import loadmat
# import mlp from scikit-learn
import numpy as np
from sklearn.neural_network import MLPClassifier

def one_hot(y):
    one_hot = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        one_hot[i][y[i][0]-1] = 1
    return one_hot

def main():
    mat = loadmat("src/nn/ex3data1.mat")
    X = mat["X"]
    y = mat["y"]
    y = one_hot(y)
    
    # Using MLPClassifier from scikit-learn
    mlp = MLPClassifier(hidden_layer_sizes=(25,), 
                        max_iter=1000, 
                        random_state=42,
                        solver='sgd', 
                        verbose=True,
                        early_stopping=False,
                        n_iter_no_change=10000,
                        batch_size=100000,
                        activation='logistic',
                        alpha=0.2, 
                        learning_rate_init=1.0,
                        momentum=0)
    mlp.fit(X, y)
    print("Training completed with scikit-learn MLPClassifier.")
    
    # plot the loss curve
    import matplotlib.pyplot as plt
    plt.plot(mlp.loss_curve_)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()
        
    # classification report
    from sklearn.metrics import classification_report
    y_pred = mlp.predict(X)
    print(classification_report(y, y_pred, target_names=[str(i) for i in range(1, 11)]))
   
    
if __name__ == "__main__":
    main()    
    