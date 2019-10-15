"""
Created by Xenia-Io @ 2019-09-20
"""

import numpy as np
from matplotlib import pyplot as plt
import random


class NeuralNetwork:

    def __init__(self, **kwargs):

        self.W = None
        self.b = None


    def initialize_params(self, K, d):
        """
        Initialize params of the model W and b

        :param d: the dimensionality of each image (3072 = 32x32x3).
        :param K: num of labels = 10
        :W: has size Kxd
        :b: is Kx1

        :return: params W and b
        """

        W = np.random.normal(0, 0.01, (K, d))
        b = np.random.normal(0, 0.01, (K, 1))

        return W, b

    def compute_softmax(self, x):

        e = x - np.max(x)
        return np.exp(e) / np.sum(np.exp(e), axis=0)


    def evaluateClassifier(self, X, W, beta):

        s = np.dot(W, X) + beta
        p = NeuralNetwork().compute_softmax(s)

        return p

    def compute_regularization(self, W, l):
        """
            compute the regularization term: l * ||W||^2
        """
        return l * np.sum(np.square(W))

    def compute_ce_loss(self, Y, P):
        """
            compute cross entropy loss
        """
        p_one_hot = np.sum(np.prod((Y, P), axis=0), axis=0)
        loss = np.sum(0 - np.log(p_one_hot))

        return loss


    def computeCost(self, X, Y, W, b, l):

        P = NeuralNetwork().evaluateClassifier(X, W, b)

        # cross entropy loss function
        loss = NeuralNetwork().compute_ce_loss(Y, P)

        # regularization term
        regularization = NeuralNetwork().compute_regularization(W, l)

        # compute cost
        n = np.shape(X)[1] # number of images
        cost = (1 / n) * loss + regularization

        return cost

    def compute_accuracy(self, X, W, b, y_labels):
        """
            Percentage of correctly classified predictions
        """

        P = NeuralNetwork().evaluateClassifier(X, W, b)
        predictions = np.argmax(P, axis=0)
        corrects = np.where(predictions - y_labels == 0)
        num_of_corrects = len(corrects[0])

        # return (num_of_corrects / np.size(y_labels[0]))

        return (num_of_corrects / X.shape[1])


    def computeGradients_analytically(self, X, Y, W, b, l):
        """
            Computes the gradients of the weight and bias parameters analytically

            Grad W is the gradient matrix of the cost J relative to W and has size K*d.
            Grad b is the gradient vector of the cost J relative to b and has size K*1.

            Note: to implement this i based on the slides 94-97/97 from lecture 3
        """
        np.seterr(divide='ignore', invalid='ignore')

        # define dimensions
        K = np.shape(Y)[0]
        d = np.shape(X)[0]
        n = np.shape(X)[1] #number of images

        # initialize gradients with zeros
        W_grad = np.zeros((K, d))
        b_grad = np.zeros((K, 1))

        P = NeuralNetwork().evaluateClassifier(X, W, b)

        # loop all the images
        for i in range(n):
            p_i = P[:, i].reshape(-1, 1)
            Y_i = Y[:, i].reshape(-1, 1)
            X_i = X[:, i].reshape(1, -1)
            g = p_i - Y_i

            b_grad += g
            W_grad += np.dot(g, X_i)

        # implement regularization
        W_grad = W_grad / n + 2 * l * W
        b_grad = b_grad / n

        return W_grad, b_grad

    def compute_gradients_num_slow(self, X, Y, W, b, l=0, h=1e-6):
        """
            Computes the gradients of the weight and bias numerically
        """
        np.seterr(divide='ignore', invalid='ignore')

        grad_w = np.zeros(np.shape(W))
        grad_b = np.zeros(np.shape(b))

        # b_try = np.copy(b)

        for i in range(np.shape(b)[0]):
            b_try = np.copy(b)
            # b = b_try
            b_try[i] = b_try[i] - h
            c1 = NeuralNetwork().computeCost(X, Y, W, b_try, l)
            b_try = np.copy(b)
            b_try[i] = b_try[i] + h
            c2 = NeuralNetwork().computeCost(X, Y, W, b_try, l)
            grad_b[i] = (c2-c1) / (2*h)

        # w_try = np.copy(W)

        for i in np.ndindex(np.shape(W)):
            w_try = np.copy(W)
            # W = w_try
            w_try[i] = w_try[i] - h
            c1 = NeuralNetwork().computeCost(X, Y, w_try, b, l)
            w_try = np.copy(W)
            w_try[i] = w_try[i] + h
            c2 = NeuralNetwork().computeCost(X, Y, w_try, b, l)
            grad_w[i] = (c2-c1) / (2*h)

        return grad_w, grad_b


    def compare_Gradients(self, X, Y, W, b, lamda, h, samples=5):

        # get dimensions
        n = np.shape(X)[1]

        random_numbers = random.sample(range(np.shape(X)[1]), samples)

        # create new objects
        X = X[100:300, random_numbers]
        Y = Y[:, random_numbers]
        W = W[:, 100:300]

        # call the methods that will be compared
        W_analytical, b_analytical = NeuralNetwork().computeGradients_analytically(X, Y, W, b, lamda)
        W_numerical, b_numerical = NeuralNetwork().compute_gradients_num_slow(X, Y, W, b, lamda, h)

        np.testing.assert_almost_equal(b_analytical, b_numerical, decimal=6)
        np.testing.assert_almost_equal(W_analytical, W_numerical, decimal=6)


    def mini_batch_gd(self, X, Y, y, X_val, Y_val, y_val, X_test, y_test, W, b, lamda, learning_rate,
                      n_batch=100, n_epochs=40, verbose=True, detailed_results_per_epoch=False):
        """
            Train the model using mini-batch gradient descent

            X             : data matrix (D, N)
            Y             : one-hot-encoding labels matrix (C, N)
            lamda         : regularization term
            n_batch       : number of batches
            learning_rate : learning rate
            n_epochs      : number of training epochs

            Returns:
                acc_train : the accuracy on the training set
                acc_val   : the accuracy on the validation set
                acc_test  : the accuracy on the testing set
        """

        train_cost = []
        train_accuracy = []
        val_cost = []
        val_accuracy = []

        # get dimensions - N is 10.000 images
        N = np.shape(X)[1]
        images_per_batch = int(N / n_batch) # 100 images per batch

        # generate the set of mini - batches and do the Gradient Descent
        for epoch in range(n_epochs):
            for batch in range(n_batch):

                j_start = (batch) * images_per_batch
                j_end = (batch + 1) * images_per_batch

                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]

                grad_W, grad_b = self.computeGradients_analytically(X_batch, Y_batch, W, b, lamda)

                W -= learning_rate * grad_W
                b -= learning_rate * grad_b

            # TRAINING
            # compute accuracy per epoch and save the results in list
            epoch_accuracy = self.compute_accuracy(X, W, b, y)
            train_accuracy.append(epoch_accuracy)

            # compute cost per epoch and save the results in list
            epoch_cost = self.computeCost(X, Y, W, b, lamda)
            train_cost.append(epoch_cost)

            # CROSS VALIDATION
            # compute accuracy per epoch and save the results in list
            val_epoch_accuracy = self.compute_accuracy(X_val, W, b, y_val)
            val_accuracy.append(val_epoch_accuracy)

            # compute cost per epoch and save the results in list
            val_epoch_cost = self.computeCost(X_val, Y_val, W, b, lamda)
            val_cost.append(val_epoch_cost)


            if detailed_results_per_epoch:
                print("At epoch ",epoch," cost is ",epoch_cost, " and accuracy is ", epoch_accuracy)


        # Test the model after training
        accuracy_train = self.compute_accuracy(X, W, b, y)
        accuracy_val = self.compute_accuracy(X_val, W, b, y_val)
        accuracy_test = self.compute_accuracy(X_test, W, b, y_test)

        if verbose:
            print("The accuracy on the training set is: ", accuracy_train)
            print("The accuracy on the validation set is: " , accuracy_val)
            print("The accuracy on the testing set is: ", accuracy_test)


        return (W, b, train_accuracy, train_cost, val_accuracy, val_cost)


    def visualization_per_epoch(self, arg1, arg2, n_epochs, ylabel, title):

        """
            Plots

            n_epochs       : number of training epochs
            accuracy_train : accuracy per epoch on the training set
            accuracy_val   : accuracy per epoch on the validation set
        """

        epochs = np.arange(n_epochs)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(epochs, arg1, label="Training set")
        ax.plot(epochs, arg2, label="Validation set")
        ax.legend()
        ax.set(xlabel='Number of epochs', ylabel = ylabel)
        ax.grid()

        plt.savefig("plots/" + title + ".png", bbox_inches="tight")

    def plot_simple_fig(self, W):
        images = []
        for i in W:
            raw_img = np.rot90(i.reshape(3, 32, 32).T, -1)
            image = ((raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img)))
            images.append(image)
        fig1 = plt.figure(figsize=(20, 5))

        return fig1, images


    def visualize_weights(self, W, title, save=True):

        fig1, images = self.plot_simple_fig(W)

        for idx in range(len(W)):
            ax = fig1.add_subplot(2, 5, idx + 1)
            ax.set_title('Class %s' % (idx + 1))
            ax.imshow(images[idx])

        if save:
            plt.savefig("plots/" + title + ".png", bbox_inches="tight")

        plt.show()
        plt.clf()