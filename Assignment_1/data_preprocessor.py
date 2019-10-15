"""
Created by Xenia-Io @ 2019-09-20
"""

import numpy as np
import pickle

num_of_classes = 10
directory = "D://Documents//KTH//SEMESTER_4//Deep_Learning//Assignment_1//Datasets//cifar-10-batches-py/"

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

    # normalization between [0,1]
    X_data = (file_data[b"data"] / 255).T

    # vector of length N containing the label for each image
    y = file_data[b"labels"]

    # one-hot representation of the label for each image
    Y_labels = (np.eye(num_of_classes)[y]).T

    return X_data, Y_labels, y


def build_dataset():
    X_train, Y_train, y_train = load_batch(directory, "/data_batch_1")
    X_val, Y_val, y_val = load_batch(directory, "/data_batch_2")
    X_test, Y_test, y_test = load_batch(directory, "/test_batch")

    return X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test


# build_dataset()