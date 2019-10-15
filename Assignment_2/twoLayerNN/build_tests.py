""" Implementation of a two-layer neural network training with
    Mini Batch Gradient Descent using Cyclical learning rate
    and a cross-entropy loss, applied to the CIFAR10 dataset.

    For the course DD2424 Deep Learning in Data Science course at KTH Royal Institute
    of Technology - October 2019
"""

__author__ = "Xenia Ioannidou"

import statistics

from neural_network import NeuralNetwork
from data_preprocessor import *
import numpy as np

file_labels = "D://Documents//KTH//SEMESTER_4//Deep_Learning//Assignment_2//twoLayerNN//Datasets//cifar-10-batches-py//batches.meta"

class Tests():

    def checking_grads(self):
        # Build dataset and find parameters
        X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test = build_dataset()
        K, D, N, C, labels = find_parameters(X_train[:30, :5], Y_train[:30, :5])

        # Build model
        net = NeuralNetwork(K, D, N, C)

        # Check gradients
        net.compare_Gradients(X_train[:30, :5], Y_train[:30, :5], 0, 5)


    def train_one_batch(self):
        """
            Build datasets where the training data consists of 10,000 images.

            Returns:
                all the separate data sets and labels (list) with correct image labels
        """
        X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test = build_dataset()

        labels = unpickle_file(file_labels)[b'label_names']

        return X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels


    def test_figure3(self):
        """Testing Figure 3"""

        # Build dataset with ONE batch
        X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = self.train_one_batch()
        K, D, N, C, labels = find_parameters(X_train, Y_train)

        # Build network
        net = NeuralNetwork(K, D, N, C)

        # Train network
        train_accuracy, train_cost,train_loss, val_accuracy, val_cost, val_loss = net.mini_batch_gd(X_train, Y_train, y_train, X_val, Y_val, y_val,
                          X_test, Y_test, y_test, lamda=0.01, learning_rate_min=1e-5, learning_rate_max=1e-1, stepsize=500,
                          batch_size=100, n_epochs=10, verbose=True, detailed_results_per_epoch=False, print_test_accuracy=False)

        # Plot cost , loss and accuracy for training and validation tests
        caption = "Costs per number of epochs for Figure 3"
        net.visualize(train_cost, val_cost, 10, 4, "Costs", "Costs per number of epochs for "+ "Figure 3", caption)
        caption = "Accuracy per number of epochs for Figure 3"
        net.visualize(train_accuracy, val_accuracy, 10, 1, "Accuracy", "Accuracy per number of epochs for "+ "Figure 3", caption)
        caption = "Loss per number of epochs for Figure 3"
        net.visualize(train_loss, val_loss, 10, 4, "Loss", "Loss per number of epochs for "+ "Figure 3", caption)


    def test_figure4(self):
        """Testing Figure 3"""

        # Build dataset with one batch
        X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = self.train_one_batch()
        K, D, N, C, labels = find_parameters(X_train, Y_train)

        # Build network
        net = NeuralNetwork(K, D, N, C)

        # Train network
        train_accuracy, train_cost, train_loss, val_accuracy, val_cost, val_loss = net.mini_batch_gd(X_train, Y_train,
                                                                                                     y_train, X_val,
                                                                                                     Y_val, y_val,
                                                                                                     X_test, Y_test,
                                                                                                     y_test, lamda=0.01,
                                                                                                     learning_rate_min=1e-5,
                                                                                                     learning_rate_max=1e-1,
                                                                                                     stepsize=800,
                                                                                                     batch_size=100,
                                                                                                     n_epochs=48,
                                                                                                     verbose=True,
                                                                                                     detailed_results_per_epoch=False,
                                                                                                     print_test_accuracy=False)

        # Plot cost , loss and accuracy for training and validation tests
        caption = "Costs per number of epochs for Figure 4"
        net.visualize(train_cost, val_cost, 48, 4, "Costs", "Costs per number of epochs for " + "Figure 4", caption)
        caption = "Accuracy per number of epochs for Figure 4"
        net.visualize(train_accuracy, val_accuracy, 48, 1, "Accuracy",
                      "Accuracy per number of epochs for " + "Figure 4", caption)
        caption = "Loss per number of epochs for Figure 4"
        net.visualize(train_loss, val_loss, 48, 4, "Loss", "Loss per number of epochs for " + "Figure 4", caption)


    def train_on_all_data_batches(self):
        """
            Create training from all images. Split data for build validation set

            Returns:
                all the separate data sets and labels (list) with correct image labels
        """
        X_train1, Y_train1, y_train1, X_train2, Y_train2, y_train2, \
        X_train3, Y_train3, y_train3, X_train4, Y_train4, y_train4, \
        X_train5, Y_train5, y_train5, X_test, Y_test, y_test = build_large_dataset()

        X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4, X_train5),
                                 axis=1)
        Y_train = np.concatenate((Y_train1, Y_train2, Y_train3, Y_train4, Y_train5),
                                 axis=1)
        y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5))

        # create validation set
        X_val = X_train[:, -1100:]
        Y_val = Y_train[:, -1100:]
        y_val = y_train[-1100:]
        X_train = X_train[:, :-1100]
        Y_train = Y_train[:, :-1100]
        y_train = y_train[:-1100]

        labels = unpickle_file(file_labels)[b'label_names']

        return X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels


    def coarse_search(self):

        # Build dataset with one batch
        X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = self.train_one_batch()
        K, D, N, C, labels = find_parameters(X_train, Y_train)

        # Build network
        net = NeuralNetwork(K, D, N, C)

        # Coarse random search of the lambda
        lamda_list = []
        while len(lamda_list) < 30:
            x = np.random.uniform(1e-3, 1e-9)
            lamda_list.append(x)

        for lamda_i in lamda_list:
            print("Lambda = ", lamda_i)

            # Train network
            train_accuracy, train_cost, train_loss, val_accuracy, val_cost, val_loss = net.mini_batch_gd(X_train,
                                                                                                         Y_train,
                                                                                                         y_train, X_val,
                                                                                                         Y_val, y_val,
                                                                                                         X_test, Y_test,
                                                                                                         y_test,
                                                                                                         lamda=lamda_i,
                                                                                                         learning_rate_min=1e-5,
                                                                                                         learning_rate_max=1e-1,
                                                                                                         stepsize=800,
                                                                                                         batch_size=100,
                                                                                                         n_epochs=20,
                                                                                                         verbose=True,
                                                                                                         detailed_results_per_epoch=False,
                                                                                                         print_test_accuracy=False)


            print("___________________________")


    def fine_search(self):

        # Build dataset with one batch
        X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = self.train_one_batch()
        K, D, N, C, labels = find_parameters(X_train, Y_train)

        # Build network
        net = NeuralNetwork(K, D, N, C)

        # Coarse random search of the lambda
        lamda_list = []
        while len(lamda_list) < 30:
            x = np.random.uniform(0.0001, 1e-6)
            lamda_list.append(x)

        for lamda_i in lamda_list:
            print("Lambda = ", lamda_i)

            # Train network
            train_accuracy, train_cost, train_loss, val_accuracy, val_cost, val_loss = net.mini_batch_gd(X_train,
                                                                                                         Y_train,
                                                                                                         y_train, X_val,
                                                                                                         Y_val, y_val,
                                                                                                         X_test, Y_test,
                                                                                                         y_test,
                                                                                                         lamda=lamda_i,
                                                                                                         learning_rate_min=1e-5,
                                                                                                         learning_rate_max=1e-1,
                                                                                                         stepsize=800,
                                                                                                         batch_size=100,
                                                                                                         n_epochs=20,
                                                                                                         verbose=True,
                                                                                                         detailed_results_per_epoch=False,
                                                                                                         print_test_accuracy=False)


            print("___________________________")


    def train_best_classifier(self):

        # Build dataset with ALL batches
        X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = self.train_on_all_data_batches()
        K, D, N, C, labels = find_parameters(X_train, Y_train)

        acc_train_set = []
        acc_val_set = []
        acc_test_set = []

        # Train network for 10 times to find statistics (mean, std) for the accuracies
        for j in range(10):
            net = NeuralNetwork(K, D, N, C)

            train_accuracy, train_cost, train_loss, val_accuracy, val_cost, val_loss = net.mini_batch_gd(X_train,
                                                                                                         Y_train,
                                                                                                         y_train, X_val,
                                                                                                         Y_val, y_val,
                                                                                                         X_test, Y_test,
                                                                                                         y_test,
                                                                                                         lamda=0.4137,
                                                                                                         learning_rate_min=1e-5,
                                                                                                         learning_rate_max=1e-1,
                                                                                                         stepsize=800,
                                                                                                         batch_size=100,
                                                                                                         n_epochs=20,
                                                                                                         verbose=True,
                                                                                                         detailed_results_per_epoch=False,
                                                                                                         print_test_accuracy=False)


            acc_train_set.append(train_accuracy)
            acc_val_set.append(val_accuracy)

        print("Train mean accuracy:" + str(statistics.mean(acc_train_set)))
        print("Validation mean accuracy:" + str(statistics.mean(acc_val_set)))
        print("Test mean accuracy:" + str(statistics.mean(acc_test_set)))
        print("Train stdev accuracy:" + str(statistics.stdev(acc_train_set)))

        np.random.seed(0)
        net = NeuralNetwork(K, D, N, C)
        train_accuracy, train_cost, train_loss, val_accuracy, val_cost, val_loss = net.mini_batch_gd( X_train, Y_train,
                                                                                                                    y_train, X_val,
                                                                                                                    Y_val, y_val,
                                                                                                                    X_test, Y_test,
                                                                                                                    y_test,
                                                                                                                    lamda=0.4137,
                                                                                                                    learning_rate_min=1e-5,
                                                                                                                    learning_rate_max=1e-1,
                                                                                                                    stepsize=800,
                                                                                                                    batch_size=100,
                                                                                                                    n_epochs=20,
                                                                                                                    verbose=False,
                                                                                                                    detailed_results_per_epoch=False,
                                                                                                                     print_test_accuracy=True)

        # Plot cost , loss and accuracy for training and validation tests
        caption = "Costs per number of epochs for Best classifier"
        net.visualize(train_cost, val_cost, 20, 4, "Costs", "Costs per number of epochs for " + "Best classifier", caption)
        caption = "Accuracy per number of epochs for Best classifier"
        net.visualize(train_accuracy, val_accuracy, 20, 1, "Accuracy",
                      "Accuracy per number of epochs for " + "Best classifier", caption)
        caption = "Loss per number of epochs for Best classifier"
        net.visualize(train_loss, val_loss, 20, 4, "Loss", "Loss per number of epochs for " + "Best classifier", caption)