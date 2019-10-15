"""
Created by Xenia-Io @ 2019-09-20
"""

from neural_network import NeuralNetwork
from data_preprocessor import build_dataset
import numpy as np

def main():

    # Build parameters
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test = build_dataset()
    K = np.shape(Y_train)[0]
    d = np.shape(X_train)[0]
    n = np.shape(X_train)[1]

    #Train and test a single network
    neural_network = NeuralNetwork()
    W, b = neural_network.initialize_params(K, d)

    lambdas = [0, 0, .1, 1]
    etas = [.1, .01, .01, .01]

    for i in range(4):
        print("Model results for lambda = ", lambdas[i], " and learning rate = ", etas[i])

        W_train, b_train, accuracy_train, cost_train, val_accuracy, val_cost = \
            neural_network.mini_batch_gd(X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, y_test,
                                         W, b, lamda = lambdas[i], learning_rate = etas[i])

        neural_network.visualization_per_epoch(cost_train, val_cost, 40, "Costs", "Costs per epoch for eta "+ str(etas[i]) +" and l = " + str(lambdas[i]))
        neural_network.visualize_weights(W_train,"case "+ str(i))

if __name__ == "__main__":
    main()