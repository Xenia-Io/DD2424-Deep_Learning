""" Implementation of a two-layer neural network training with
    Mini Batch Gradient Descent using Cyclical learning rate
    and a cross-entropy loss, applied to the CIFAR10 dataset.

    For the course DD2424 Deep Learning in Data Science course at KTH Royal Institute
    of Technology - October 2019
"""

__author__ = "Xenia Ioannidou"

import numpy as np
import pickle

num_of_classes = 10
directory = "D://Documents//KTH//SEMESTER_4//Deep_Learning//Assignment_2//twoLayerNN//Datasets//cifar-10-batches-py/"
file_labels = "D://Documents//KTH//SEMESTER_4//Deep_Learning//Assignment_2//twoLayerNN//Datasets//cifar-10-batches-py//batches.meta"


def unpickle_file(filename):

    # serialise the object before writing it to file:
    # convert it into a character stream
    with open(filename, 'rb') as file_opened:
        file_data = pickle.load(file_opened, encoding='bytes')

    return file_data


def load_batch(directory, extension):
    """
        dataset:
            60.000 32x32 colour images in 10 classes, with 6.000 images per class

            5 training batches: 1 batch has 10.000 images, so 50.000 training images and
            5 training batches: 5000 images from each class, 10 classes, so 50.000 training images
            1 test batch: 10.000 test images
            1 image is 32x32x3

        labels:
            a list of 10.000 numbers in the range 0-9
            the number in position i is the label of the i_th image in the array data

        A.data:
            an array 10.000 rows and 3072 columns (32x32x3), where 1 image is one row in A.data

        return:
            the image and label data in separate files
    """

    file = directory + extension

    data = unpickle_file(file)

    X, Y, y = preprocess_data(data, num_of_classes)

    return X, Y, y


def preprocess_data(file_data, num_of_classes):
    """
        :param file_data: normalizing between [0,1]
        :X_data: contains the image pixel data which is d x N = (3072, 10000)
        :y: vector of length N = 10.000 containing the label for each image
        :Y is KxN (K = 10) and contains the one-hot representation of the label for each image
        :return: data and labels
    """

    # normalization with respect to the mean and std
    # 1 column in X is 1 image, so mean and std is per column
    X_data = (file_data[b"data"]).T
    X_mean = np.mean(X_data, axis=1, keepdims=True)
    X_stdev = np.std(X_data, axis=1, keepdims=True)

    X_data = (X_data - X_mean) / X_stdev

    # the label for each image - vector of length N
    y = np.asarray(file_data[b"labels"])

    # one-hot representation of the label for each image
    Y = (np.eye(num_of_classes)[y]).T

    return X_data, Y, y


def build_dataset():
    X_train, Y_train, y_train = load_batch(directory, "/data_batch_1")
    X_val, Y_val, y_val = load_batch(directory, "/data_batch_2")
    X_test, Y_test, y_test = load_batch(directory, "/test_batch")

    print("X_train shape = ", X_train.shape)
    print("Y_train shape = ", Y_train.shape)

    print("X_val shape = ", X_val.shape)
    print("Y_val shape = ", Y_val.shape)

    print("X_test shape = ", X_test.shape)
    print("Y_test shape = ", Y_test.shape)

    return X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test


def build_large_dataset():

    X_train1, Y_train1, y_train1 = load_batch(directory, "/data_batch_1")
    X_train2, Y_train2, y_train2 = load_batch(directory, "/data_batch_2")
    X_train3, Y_train3, y_train3 = load_batch(directory, "/data_batch_3")
    X_train4, Y_train4, y_train4 = load_batch(directory, "/data_batch_4")
    X_train5, Y_train5, y_train5 = load_batch(directory, "/data_batch_5")

    X_test, Y_test, y_test = load_batch(directory, "/test_batch")

    return X_train1, Y_train1, y_train1, X_train2, Y_train2, y_train2,\
           X_train3, Y_train3, y_train3, X_train4, Y_train4, y_train4, \
           X_train5, Y_train5, y_train5, X_test, Y_test, y_test



def find_parameters(X, Y):
    K = np.shape(Y)[0]
    D = np.shape(X)[0]
    N = np.shape(X)[1]

    labels = unpickle_file(file_labels)[b'label_names']

    C = len(labels)

    return K, D, N, C, labels

# build_dataset()


