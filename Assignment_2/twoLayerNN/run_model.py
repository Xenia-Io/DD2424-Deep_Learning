""" Implementation of a two-layer neural network training with
    Mini Batch Gradient Descent using Cyclical learning rate
    and a cross-entropy loss, applied to the CIFAR10 dataset.

    For the course DD2424 Deep Learning in Data Science course at KTH Royal Institute
    of Technology - October 2019
"""

__author__ = "Xenia Ioannidou"

from neural_network import NeuralNetwork
from data_preprocessor import *
import numpy as np
from build_tests import *


def main():

    tester = Tests()

    # Exercise 1, 2
    # tester.checking_grads()

    # Exercise 3_a
    # tester.test_figure3()
    tester.test_figure4()

    # Exercise 3_b
    # tester.coarse_search()
    # tester.fine_search()

    # Exercise 4
    # tester.train_best_classifier()


if __name__ == "__main__":
    main()