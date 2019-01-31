'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from data_helpers import load_data
from run_classifiers import generate_dirs
import os


class CNN(object):
    def __init__(self):
        self.filters = 250
        self.kernel_size = 3
        self.hidden_dims = 250
        self.pretrained = True
        self.W = []
        self.directory = "../results/models/"

    def run(self, max_len, batch_size, epochs, verbose, data):
        x_train, x_test, y_train, y_test = data
        self.input_length = max_len

        if verbose:
            print(len(x_train), 'train sequences')
            print(len(x_test), 'test sequences')
            print('Pad sequences (samples x time)')
            print(self.input_length, "maximal length")
        if verbose:
            print('x_train shape:', x_train.shape)
            print('x_test shape:', x_test.shape)

            print('Build model...')

        model = Sequential()


        if self.pretrained:
            vocab_size = self.W.shape[0]
            embedding_size = self.W.shape[1]
            model.add(Embedding(vocab_size, embedding_size, weights=[self.W],
                                input_length=self.input_length))
            model.add(Dropout(0.2))
        else:
            # we start off with an efficient embedding layer which maps
            # our vocab indices into embedding_dims dimensions

            embedding_dims = 300
            max_features = 1000
            model.add(Embedding(max_features,
                                embedding_dims,
                                input_length=self.input_length))



        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        model.add(Conv1D(self.filters, self.kernel_size,
                         padding='valid', activation='relu', strides=1))
        # we use max pooling:
        model.add(GlobalMaxPooling1D())

        # We add a vanilla hidden layer:
        model.add(Dense(self.hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))

        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        # plot_model(model, to_file='Sent_FF.png', show_shapes=False, show_layer_names=True, rankdir='TB')

        # loss function = binary_crossentropy
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        if verbose:
            print(model.summary())
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            validation_data=(x_test, y_test))

        self.score, self.acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=verbose)
        if verbose:
            print('Test score:', self.score)
            print('Test accuracy:', self.acc)

        home = self.directory
        model.save(home + "cnn.h5")


if __name__ == '__main__':
    # set parameters:
    this_batch_size = 32
    this_epochs = 5
    verbose = False
    this_data_directory, this_results_directory = generate_dirs()
    this_emb_file = this_data_directory + "GoogleNews-vectors-negative300.bin"
    this_emb_set = "w2v"

    this_train = this_data_directory + "task1.train.txt"
    this_data, this_input_length, this_W, t = load_data(this_train,
                                                        this_emb_file,
                                                        this_emb_set,
                                                        True)

    clf = CNN()
    clf.W = this_W
    clf.pretrained = True
    results_dir = this_results_directory + this_emb_file + "/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    clf.directory = results_dir
    clf.run(this_input_length, this_batch_size, this_epochs, verbose, this_data)
    print(clf.score, clf.acc)
