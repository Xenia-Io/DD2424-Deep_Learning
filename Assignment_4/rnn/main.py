"""
Created by Xenia-Io @ 2021-05-03

Implementation of a vanilla recurrent neural network
to synthesize English text character by character.

Assignment 4 of the DD2424 Deep Learning in Data Science course at
KTH Royal Institute of Technology
"""

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import random


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
            data (dict) with the following columns:
              - book_data   : the text as a long string (str)
              - book_chars  : dict of unique characters in the text (book_data)
              - vocab_len   : vocabulary size
              - char_to_ind : mapping of characters to indices
              - ind_to_char : mapping of indices to characters
        """
        book_data = open(self.filename, 'r', encoding='utf8').read()
        book_chars = list(set(book_data))

        data = {"book_data": book_data,
                "book_chars": book_chars,
                "vocab_len": len(book_chars),
                "char_to_ind": OrderedDict((char, idx) for idx, char in
                                           enumerate(book_chars)),
                "ind_to_char": OrderedDict((idx, char) for idx, char in
                                           enumerate(book_chars))}

        return data


class RNN():
    """A vanilla RNN model"""

    def __init__(self, data, m=100, eta=0.1, seq_length=25, sigma= 0.01):
        """
        Build the RNN model

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
            data (dict)     : get info from the loaded text file
        """

        self.m, self.eta, self.seq_length = m, eta, seq_length
        self.vocab_len = data['vocab_len']
        self.ind_to_char = data['ind_to_char']
        self.char_to_ind = data['char_to_ind']
        self.book_data = data['book_data']

        self.b = np.zeros((m, 1))
        self.c = np.zeros((self.vocab_len, 1))

        self.U = np.random.normal(0, sigma, size=(m, self.vocab_len))
        self.W = np.random.normal(0, sigma, size=(m, m))
        self.V = np.random.normal(0, sigma, size=(self.vocab_len, m))


    def compute_softmax(self, x):
        e = x - np.max(x)
        return np.exp(e) / np.sum(np.exp(e), axis=0)


    def evaluate_classifier(self, h, x):
        """
        Forward-pass of the classifier

        Args:
            h (np.ndarray): hidden state sequence
            X (np.ndarray): sequence of input vectors, where each
                            x has size (dx1)
        Returns:
            a (np.ndarray): linear transformation of W and U + bias b
            h (np.ndarray): tanh activation of a
            o (np.ndarray): linear transformation of V + bias c
            p (np.ndarray): softmax activation of o
        """
        a = np.matmul(self.W, h) + np.matmul(self.U, x) + self.b
        h = np.tanh(a)
        o = np.matmul(self.V, h) + self.c
        p = self.compute_softmax(o)

        return a, h, o, p


    def synthesize_text(self, h, ix, n):
        """
        Generate text based on the hidden state sequence
        Input vectors are one-hot-encoded

        Args:
            n     (int)        : length of the sequence to be generated
            h0    (np.ndarray) : hidden state at time 0
            idx   (np.ndarray) : index of the first dummy input vector

        Returns:
            text (str): a synthesized string of length n
        """
        # The next input vector
        xnext = np.zeros((self.vocab_len, 1))
        # Use the index to set the net input vector
        xnext[ix] = 1 # 1-hot-encoding

        txt = ''
        for t in range(n):
            _, h, _, p = self.evaluate_classifier(h, xnext)
            # At each time step t when you generate a
            # vector of probabilities for the labels,
            # you then have to sample a label from this PMF
            ix = np.random.choice(range(self.vocab_len), p=p.flat)
            xnext = np.zeros((self.vocab_len, 1))
            xnext[ix] = 1 # Lecture 9, page 22
            txt += self.ind_to_char[ix]

        return txt


    def compute_gradients(self, inputs, targets, hprev):
        """
           Analytically computes the gradients of the weight and bias parameters
        """
        n = len(inputs)
        loss = 0

        # Dictionaries for storing values during the forward pass
        aa, xx, hh, oo, pp = {}, {}, {}, {}, {}
        hh[-1] = np.copy(hprev)

        # Forward pass
        for t in range(n):
            xx[t] = np.zeros((self.vocab_len, 1))
            xx[t][inputs[t]] = 1 # 1-hot-encoding

            aa[t], hh[t], oo[t], pp[t] = self.evaluate_classifier(hh[t-1], xx[t])

            loss += -np.log(pp[t][targets[t]][0]) # update the loss

        # Dictionary for storing the gradients
        grads = {"W": np.zeros_like(self.W), "U": np.zeros_like(self.U),
                 "V": np.zeros_like(self.V), "b": np.zeros_like(self.b),
                 "c": np.zeros_like(self.c), "o": np.zeros_like(pp[0]),
                 "h": np.zeros_like(hh[0]), "h_next": np.zeros_like(hh[0]),
                 "a": np.zeros_like(aa[0])}

        # Backward pass
        for t in reversed(range(n)):
            grads["o"] = np.copy(pp[t])
            grads["o"][targets[t]] -= 1

            grads["V"] += grads["o"]@hh[t].T
            grads["c"] += grads["o"]

            grads["h"] = np.matmul(self.V.T , grads["o"] )+ grads["h_next"]
            grads["a"] = np.multiply(grads["h"], (1 - np.square(hh[t])))

            grads["U"] += np.matmul(grads["a"], xx[t].T)
            grads["W"] += np.matmul(grads["a"], hh[t-1].T)
            grads["b"] += grads["a"]

            grads["h_next"] = np.matmul(self.W.T, grads["a"])

        # Drop redundant gradients
        grads = {k: grads[k] for k in grads if k not in ["o", "h", "h_next", "a"]}

        # Clip the gradients
        for grad in grads:
            grads[grad] = np.clip(grads[grad], -5, 5)

        # Update the hidden state sequence
        h = hh[n-1]

        return grads, loss, h


    def compute_gradients_num(self, inputs, targets, hprev, h, num_comps=20):
        """
           Numerically computes the gradients of the weight and bias parameters
        """
        rnn_params = {"W": self.W, "U": self.U, "V": self.V, "b": self.b, "c": self.c}
        num_grads  = {"W": np.zeros_like(self.W), "U": np.zeros_like(self.U),
                      "V": np.zeros_like(self.V), "b": np.zeros_like(self.b),
                      "c": np.zeros_like(self.c)}

        for key in rnn_params:
            for i in range(num_comps):
                old_par = rnn_params[key].flat[i] # store old parameter
                rnn_params[key].flat[i] = old_par + h
                _, l1, _ = self.compute_gradients(inputs, targets, hprev)
                rnn_params[key].flat[i] = old_par - h
                _, l2, _ = self.compute_gradients(inputs, targets, hprev)
                rnn_params[key].flat[i] = old_par # reset parameter to old value
                num_grads[key].flat[i] = (l1 - l2) / (2*h)

        return num_grads


    def check_gradients(self, inputs, targets, hprev, num_comps=20):
        """
           Check similarity between the analytical and numerical gradients

        """
        grads_ana, _, _ = self.compute_gradients(inputs, targets, hprev)
        grads_num = self.compute_gradients_num(inputs, targets, hprev, 1e-5)

        print("Gradient checks:")
        for grad in grads_ana:
            num   = abs(grads_ana[grad].flat[:num_comps] -
                    grads_num[grad].flat[:num_comps])
            denom = np.asarray([max(abs(a), abs(b)) + 1e-10 for a,b in
                zip(grads_ana[grad].flat[:num_comps],
                    grads_num[grad].flat[:num_comps])
            ])
            max_rel_error = max(num / denom)

            print("The maximum relative error for the %s gradient is: %e." %
                    (grad, max_rel_error))
        print()


def main():
    # Book position tracker, iteration, epoch
    e, n, epoch = 0, 0, 0
    num_epochs = 30

    smooth_loss_lst = []

    # Load dataset
    loader = DataLoader("goblet_book.txt")
    data = loader.load_dataset()

    # Build model
    rnn = RNN(data)

    rnn_params = {"W": rnn.W, "U": rnn.U, "V": rnn.V, "b": rnn.b, "c": rnn.c}

    mem_params = {"W": np.zeros_like(rnn.W), "U": np.zeros_like(rnn.U),
                  "V": np.zeros_like(rnn.V), "b": np.zeros_like(rnn.b),
                  "c": np.zeros_like(rnn.c)}

    while epoch < num_epochs:

        # Re-initialization
        if n == 0 or e >= (len(rnn.book_data) - rnn.vocab_len - 1):
            if epoch != 0: print("Finished %i epochs." % epoch)
            hprev = np.zeros((rnn.m, 1))
            e = 0
            epoch += 1

        inputs = [rnn.char_to_ind[char] for char in rnn.book_data[e:e+rnn.vocab_len]]
        targets = [rnn.char_to_ind[char] for char in rnn.book_data[e+1:e+rnn.vocab_len+1]]

        grads, loss, hprev = rnn.compute_gradients(inputs, targets, hprev)

        # Compute the smooth loss
        if n == 0 and epoch == 1:
            smooth_loss = loss
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss
        smooth_loss_lst.append(smooth_loss)

        # Check gradients
        if n == 0:
            rnn.check_gradients(inputs, targets, hprev)

        # Print the loss
        if n % 100 == 0:
            print('Iteration %d, smooth loss: %f' % (n, smooth_loss))

        # Print synthesized text
        if n % 500 == 0:
            txt = rnn.synthesize_text(hprev, inputs[0], 200)
            print('\nSynthesized text after %i iterations:\n %s\n' % (n, txt))
            print('Smooth loss: %f' % smooth_loss)

        # Adagrad
        for key in rnn_params:
            mem_params[key] += grads[key] * grads[key]
            rnn_params[key] -= rnn.eta / np.sqrt(mem_params[key] +
                    np.finfo(float).eps) * grads[key]

        e += rnn.vocab_len
        n += 1

    # Plot smooth loss
    plt.plot(smooth_loss_lst)
    plt.ylabel('Smooth Loss')
    plt.xlabel('Iterations')
    plt.show()

if __name__ == '__main__':
    main()