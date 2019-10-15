""" Implementation of a two-layer neural network training with
    Mini Batch Gradient Descent using Cyclical learning rate
    and a cross-entropy loss, applied to the CIFAR10 dataset.

    For the course DD2424 Deep Learning in Data Science course at KTH Royal Institute
    of Technology - October 2019
"""

__author__ = "Xenia Ioannidou"

import numpy as np
from data_preprocessor import *
import matplotlib.pyplot as plt


class NeuralNetwork():
    """Mini-batch gradient descent classifier"""

    def __init__(self, K, D, N, C, M=50, W1=None, b1=None, W2=None, b2=None):
        """
            M  : number of nodes in hidden layer
            W1 : weight matrix (M, D)
            b1 : bias matrix (M, 1)
            W2 : weight matrix (C, M)
            b2 : bias matrix (C, 1)
        """

        self.M = M

        if W1 != None:
            self.W1 = W1
        else:
            self.W1 = np.random.normal(0, 1 / np.sqrt(D), (M, D))

        if b1 != None:
            self.b1 = b1
        else:
            self.b1 = np.zeros((M, 1))

        if W2 != None:
            self.W2 = W2
        else:
            self.W2 = np.random.normal(0, 1 / np.sqrt(M), (C, M))

        if b2 != None:
            self.b2 = b2
        else:
            self.b2 = np.zeros((C, 1))


    def compute_softmax(self, x):

        e = x - np.max(x)
        return np.exp(e) / np.sum(np.exp(e), axis=0)

    def compute_relu(self, x):

        result = np.maximum(x, 0)
        return result

    def evaluateClassifier(self, X):
        """
            Find output of the classifier by applying ReLU on the 1st layer
               and softmax on the 2nd layer
        """
        h = self.compute_relu(np.dot(self.W1, X) + self.b1)
        p = self.compute_softmax(np.dot(self.W2, h) + self.b2)

        return h, p


    def compute_regularization(self, W1, W2, l):
        """
            compute the regularization term
        """

        return l * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    def compute_ce_loss(self, Y, P):
        """
            compute cross entropy loss
        """

        p_one_hot = np.sum(np.prod((Y, P), axis=0), axis=0)
        loss = np.sum(0 - np.log(p_one_hot))

        return loss

    def computeCost(self, X, Y, lamda):
        """
            Find the cost of the neural network

            X    : array with (D, N) dimensions
            Y    : one-hot encoding labels array (C, N)
            lamda: the regularization term
        """

        H, P = self.evaluateClassifier(X)

        # cross entropy loss function
        loss = self.compute_ce_loss(Y, P) / np.shape(X)[1]

        # regularization term
        regularization = self.compute_regularization(self.W1, self.W2, lamda)

        # compute cost
        cost = loss + regularization

        return cost, loss

    def compute_accuracy(self, X, y):
        """
            Computes the accuracy of the classifier

            X : data matrix (D, N)
            y : labels vector (N)
        """

        P = self.evaluateClassifier(X)
        predictions = np.argmax(P[1], axis=0)
        corrects = np.where(predictions - y == 0)
        num_of_corrects = len(corrects[0])

        # return (num_of_corrects / np.size(y_labels[0]))

        return (num_of_corrects / X.shape[1])


    def is_index_positive(self, x):
        if x>0:
            above_zero_indices = x
            x[above_zero_indices] = 1
            return x
        else:
            below_zero_indices = x
            x[below_zero_indices] = 0
            return x


    def compute_gradients(self, X_batch, Y_batch, labda):
        """
            Analyticcaly computes the gradients of the weight and bias parameters

            X_batch : batch matrix (D, N)
            Y_batch : one-hot-encoding labels batch vector (C, N)
            labda   : regularization term
        """
        N = X_batch.shape[1]

        # Forward pass
        H_batch, P_batch = self.evaluateClassifier(X_batch)

        # Backward pass
        G_batch = P_batch - Y_batch

        grad_W2 = ((np.dot(G_batch , np.transpose(H_batch))) / N )+ 2 * labda * self.W2
        grad_b2 = np.reshape(1 / N * np.dot(G_batch , np.ones(N)), (Y_batch.shape[0], 1))

        G_batch = np.dot(np.transpose(self.W2) , G_batch)
        H_batch[H_batch <= 0] = 0

        # returns G_batch if H_batch > 0
        G_batch = np.multiply(G_batch, H_batch > 0)

        grad_W1 = ((np.dot(G_batch , np.transpose(X_batch))) / N) + labda * self.W1
        grad_b1 = np.reshape( (np.dot(G_batch , np.ones(N))) / N, (self.M, 1))

        return grad_W1, grad_b1, grad_W2, grad_b2


    def compute_gradients_num(self, X_batch, Y_batch, labda=0, h=1e-7):
        """
            Numerically computes the gradients of the weight and bias parameters
        """
        grads = {}
        for j in range(1, 3):
            selfW = getattr(self, 'W' + str(j))
            selfB = getattr(self, 'b' + str(j))
            grads['W' + str(j)] = np.zeros(selfW.shape)
            grads['b' + str(j)] = np.zeros(selfB.shape)

            b_try = np.copy(selfB)
            for i in range(selfB.shape[0]):
                selfB = b_try[:]
                selfB[j] = selfB[j] + h
                c2, l2 = self.computeCost(X_batch, Y_batch, labda)
                getattr(self, 'b' + str(j))[i] = getattr(self, 'b' + str(j))[i] - 2 * h
                c3, l3 = self.computeCost(X_batch, Y_batch, labda)
                grads['b' + str(j)][i] = (c2 - c3) / (2 * h)

            W_try = np.copy(selfW)
            for i in np.ndindex(selfW.shape):
                selfW = W_try[:, :]
                selfW[i] = selfW[i] + h
                c2, l2 = self.computeCost(X_batch, Y_batch, labda)
                getattr(self, 'W' + str(j))[i] = getattr(self, 'W' + str(j))[i] - 2 * h
                c3, l3 = self.computeCost(X_batch, Y_batch, labda)
                grads['W' + str(j)][i] = (c2 - c3) / (2 * h)

        return grads['W1'], grads['b1'], grads['W2'], grads['b2']


    def compare_Gradients(self, X, Y, lamda, decimal):

        grad_W1_ana, grad_b1_ana, grad_W2_ana, grad_b2_ana = self.compute_gradients(X, Y, lamda)

        grad_W1_num, grad_b1_num, grad_W2_num, grad_b2_num = self.compute_gradients_num(X, Y, lamda)

        np.testing.assert_almost_equal(grad_W1_ana, grad_W1_num, decimal)
        np.testing.assert_almost_equal(grad_W2_ana, grad_W2_num, decimal)
        np.testing.assert_almost_equal(grad_b1_ana, grad_b1_num, decimal)
        np.testing.assert_almost_equal(grad_b2_ana, grad_b2_num, decimal)

    def cyclical_learning_rate(self, learning_rate, stepsize, learning_rate_min, learning_rate_max, t):

        if t <= stepsize:
            learning_rate = learning_rate_min + t / stepsize * (learning_rate_max - learning_rate_min)

        elif t <= 2 * stepsize:
            learning_rate = learning_rate_max - (t - stepsize) / stepsize * (
                        learning_rate_max - learning_rate_min)

        return learning_rate


    def mini_batch_gd(self, X, Y, y, X_val, Y_val, y_val, X_test, Y_test, y_test, lamda, learning_rate_min, learning_rate_max, stepsize,
                          batch_size, n_epochs, verbose, detailed_results_per_epoch, print_test_accuracy):
        """
            TRAIN the model using mini-batch gradient descent
        """


        train_cost = []
        train_loss = []
        train_accuracy = []
        val_cost = []
        val_loss = []
        val_accuracy = []

        # get dimensions - N is 10.000 images
        N = np.shape(X)[1]
        n_batch = int(np.floor(X.shape[1] / batch_size))
        images_per_batch = int(N / n_batch)
        # images_per_batch = int(N / batch_size)

        learning_rate_current = learning_rate_min
        t = 0

        # generate the set of mini - batches and do the Gradient Descent
        for epoch in range(n_epochs):
            for batch in range(n_batch):

                j_start = (batch) * images_per_batch
                j_end = (batch + 1) * images_per_batch

                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]

                grad_W1, grad_b1, grad_W2, grad_b2 = self.compute_gradients(X_batch, Y_batch, lamda)

                # Update parameters
                self.W1 -= learning_rate_current * grad_W1
                self.b1 -= learning_rate_current * grad_b1
                self.W2 -= learning_rate_current * grad_W2
                self.b2 -= learning_rate_current * grad_b2

                # Apply cyclical learning rates
                learning_rate_current = self.cyclical_learning_rate(learning_rate_current, stepsize, learning_rate_min, learning_rate_max, t)

                t = (t + 1) % (2 * stepsize)

            # TRAINING
            # compute accuracy per epoch and save the results in list
            epoch_accuracy = self.compute_accuracy(X, y)
            train_accuracy.append(epoch_accuracy)

            # compute cost per epoch and save the results in list
            epoch_cost, epoch_loss = self.computeCost(X, Y, lamda)
            train_cost.append(epoch_cost)
            train_loss.append(epoch_loss)

            # CROSS VALIDATION
            # compute accuracy per epoch and save the results in list
            val_epoch_accuracy = self.compute_accuracy(X_val, y_val)
            val_accuracy.append(val_epoch_accuracy)

            # compute cost per epoch and save the results in list
            val_epoch_cost, val_epoch_loss = self.computeCost(X_val, Y_val, lamda)
            val_cost.append(val_epoch_cost)
            val_loss.append(val_epoch_loss)


            if detailed_results_per_epoch:
                print("At epoch ",epoch," cost is ",epoch_cost, " and accuracy is ", epoch_accuracy)


        # TEST the model after training
        accuracy_train = self.compute_accuracy(X, y)
        accuracy_val = self.compute_accuracy(X_val, y_val)
        accuracy_test = self.compute_accuracy(X_test, y_test)

        if verbose:
            print("The accuracy on the training set is: ", accuracy_train)
            print("The accuracy on the validation set is: " , accuracy_val)
            print("The accuracy on the testing set is: ", accuracy_test)

        if print_test_accuracy:
            print("The accuracy on the testing set is: ", accuracy_test)

        return (train_accuracy, train_cost, train_loss, val_accuracy, val_cost, val_loss)


    def visualize(self, arg1, arg2, n_epochs, y_limit, ylabel, title, caption):

        epochs = np.arange(n_epochs)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(epochs, arg1, label="Training set")
        ax.plot(epochs, arg2, label="Validation set")
        ax.legend()
        ax.set(xlabel='Number of epochs', ylabel = ylabel)
        ax.set_ylim([0, y_limit])
        ax.grid()
        ax.set_title(caption)

        plt.savefig("plots/" + title + ".png", bbox_inches="tight")
