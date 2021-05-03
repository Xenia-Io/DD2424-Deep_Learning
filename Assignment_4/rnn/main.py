"""
Created by Xenia-Io @ 2021-05-03

Implementation of a vanilla recurrent neural network
to synthesize English text character by character.

Assignment 4 of the DD2424 Deep Learning in Data Science course at
KTH Royal Institute of Technology
"""

__author__ = "Xenia Ioannidou"

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import random
import sys


class DataLoader():
    """A class that loads and process the dataset"""

    def __init__(self, filename):
        """Load the dataset
        Args:
            filename (str): filename of the text to be loaded
        """
        self.filename = filename

    def load_dataset(self):
        """Load a text file and preprocess it for the RNN model

        Returns:
            data (dataframe) with the following columns:
              - book_data   : the text as a long string (str)
              - book_chars  : dict of unique characters in the text (book_data)
              - vocab_len   : vocabulary size
              - char_to_ind : mapping of characters to indices
              - ind_to_char : mapping of indices to characters
        """
        book_data = open(self.filename, 'r', encoding='utf8').read()
        book_chars = list(set(book_data))
        print("book_chars = ",
              book_chars)

        data = {"book_data": book_data,
                "book_chars": book_chars,
                "vocab_len": len(book_chars),
                "char_to_ind": OrderedDict((char, idx) for idx, char in
                                           enumerate(book_chars)),
                "ind_to_char": OrderedDict((idx, char) for idx, char in
                                           enumerate(book_chars))}

        print(data['vocab_len'])
        print(data['char_to_ind'])
        print(data['ind_to_char'])

        return data


class RNN():
    """A vanilla RNN model"""

    def __init__(self, data, m=100, eta=0.1, seq_length=25, sigma=0.01):
        """ Build the RNN model
        Args:
            b (np.ndarray)  : bias vector of length (m x 1)
            c (np.ndarray)  : bias vector of length (K x 1)
            U (np.ndarray)  : input-to-hidden weight matrix of shape (m x K)
            V (np.ndarray)  : hidden-to-output weight matrix of shape (K x m)
            W (np.ndarray)  : hidden-to-hidden weight matrix of shape (m x m)
            m (int)         : dimensionality of hidden state
            eta (float)     : learning rate
            seq_length(int) : the length of the sequence that the model uses to
                              traverse the text
        """
        self.m, self.eta, self.seq_length = m, eta, seq_length
        self.vocab_len = data['vocab_len']
        self.b = np.zeros((m, 1))
        self.c = np.zeros((self.vocab_len, 1))

        self.U = np.random.normal(0, sigma, size=(m, self.vocab_len))
        self.W = np.random.normal(0, sigma, size=(m, m))
        self.V = np.random.normal(0, sigma, size=(self.vocab_len, m))


    def compute_softmax(self, x):
        e = x - np.max(x)
        return np.exp(e) / np.sum(np.exp(e), axis=0)

    def evaluate_classifier(self, h, x):
        """ Evaluate the classifier
        Args:
            h (np.ndarray): hidden state sequence
            x (np.ndarray): sequence of input vectors
        Returns:
            a (np.ndarray): linear transformation of W and U + bias b
            h (np.ndarray): tanh activation of a
            o (np.ndarray): linear transformation of V + bias c
            p (np.ndarray): softmax activation of o
        """
        a = np.dot(self.W, h) + np.dot(self.U, x) + self.b
        h = np.tanh(a)
        o = np.dot(self.V, h) + self.c
        p = self.compute_softmax(o)

        return a, h, o, p


if __name__ == '__main__':
    loader = DataLoader("goblet_book.txt")
    data = loader.load_dataset()

    rnn = RNN(data)


