'''This example demonstrates the use of fasttext for text classification

Based on Joulin et al's paper:

Bags of Tricks for Efficient Text Classification
https://arxiv.org/abs/1607.01759

Results on IMDB datasets with uni and bi-gram embeddings:
    Uni-gram: 0.8813 test accuracy after 5 epochs. 8s/epoch on i7 cpu.
    Bi-gram : 0.9056 test accuracy after 5 epochs. 2s/epoch on GTx 980M gpu.
'''

from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from data_helpers import load_data, load_bin_vec, get_W


class fastText(object):

    def __init__(self):
        self.ngram_range = 2
        self.pretrained = False
        self.weights = []

    def create_ngram_set(self, input_list, ngram_value=2):
        """
        Extract a set of n-grams from a list of integers.

        create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
        {(4, 9), (4, 1), (1, 4), (9, 4)}

        create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
        [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
        """
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))


    def add_ngram(self, sequences, token_indice, ngram_range=2):
        """
        Augment the input list of list (sequences) by appending n-grams values.

        Example: adding bi-gram
        sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
        add_ngram(sequences, token_indice, ngram_range=2)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

        Example: adding tri-gram
        sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
        add_ngram(sequences, token_indice, ngram_range=3)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
        """
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for ngram_value in range(2, ngram_range + 1):
                for i in range(len(new_list) - ngram_value + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)

        return new_sequences

    # Set parameters:
    # ngram_range = 2 will add bi-grams features

    def run(self, max_len, batch_size, epochs, verbose, data):
        x_train, x_test, y_train, y_test = data
        max_features = max_len
        if verbose:
            print('Average train sequence length: {}'.format(
            np.mean(list(map(len, x_train)), dtype=int)))
            print('Average test sequence length: {}'.format(
            np.mean(list(map(len, x_test)), dtype=int)))

        if self.ngram_range > 1:
            if verbose:
                print('Adding {}-gram features'.format(self.ngram_range))
            # Create set of unique n-gram from the training set.
            ngram_set = set()
            for input_list in x_train:
                for i in range(2, self.ngram_range + 1):
                    set_of_ngram = self.create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

            # Dictionary mapping n-gram token to a unique integer.
            # Integer values are greater than max_features in order
            # to avoid collision with existing features.
            start_index = max_features + 1
            token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
            indice_token = {token_indice[k]: k for k in token_indice}

            # max_features is the highest integer that could be found in the dataset.
            max_features = np.max(list(indice_token.keys())) + 1

            # Augmenting x_train and x_test with n-grams features
            x_train = self.add_ngram(x_train, token_indice, self.ngram_range)
            x_test = self.add_ngram(x_test, token_indice, self.ngram_range)

            if verbose:
                print('Average train sequence length: {}'.format(
                    np.mean(list(map(len, x_train)), dtype=int)))
                print('Average test sequence length: {}'.format(
                np.mean(list(map(len, x_test)), dtype=int)))

        if verbose:
            print('Pad sequences (samples x time)')
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        if verbose:
            print('x_train shape:', x_train.shape)
            print('x_test shape:', x_test.shape)

        if verbose:
            print('Build model...')
        model = Sequential()

        if self.pretrained:
            vocab_size = self.weights.shape[0]
            embedding_size = self.weights.shape[1]
            model.add(Embedding(vocab_size, embedding_size,
                                weights=[self.weights],
                                input_length=maxlen))
        else:
            # we start off with an efficient embedding layer which maps
            # our vocab indices into embedding_dims dimensions
            model.add(Embedding(max_features,
                                embedding_dims,
                                input_length=maxlen))


        # we add a GlobalAveragePooling1D, which will average the embeddings
        # of all words in the document
        model.add(GlobalAveragePooling1D())

        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            validation_data=(x_test, y_test))

        self.score, self.acc = model.evaluate(x_test, y_test, verbose=verbose,
                                              batch_size=batch_size)

        if verbose:
            print('Test score:', self.score)
            print('Test accuracy:', self.acc)


if __name__ == '__main__':
    ngram_range = 2
    max_features = 1000
    maxlen = 400
    batch_size = 32
    embedding_dims = 300
    epochs = 20
    verbose = False
    raw_data = "./data/yelp_labelled.txt"
    data = load_data(raw_data)
    vectors = load_bin_vec("./data/GoogleNews-vectors-negative300.bin", data[-1])
    W, word_idx_map = get_W(vectors, embedding_dims)
    clf = fastText()
    clf.weights = W
    clf.pretrained = True
    clf.run(max_features, maxlen, batch_size, epochs, verbose, data)

