"""
Created by Xenia-Io @ 2021-05-03

Implementation of a vanilla recurrent neural network
to synthesize English text character by character.

Assignment 4 of the DD2424 Deep Learning in Data Science course at
KTH Royal Institute of Technology
"""

__author__ = "Xenia Ioannidou"

import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
import pandas as pd


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
              - book_data   : the text
              - book_chars  : dict of unique characters in the text
              - vocab_len   : vocabulary size
              - char_to_ind : mapping of characters to indices
              - ind_to_char : mapping of indices to characters
        """
        print(self.filename)

if __name__ == '__main__':
    loader = DataLoader("goblet_book.txt")
    loader.load_dataset()


