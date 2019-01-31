'''Train a recurrent convolutional network on the IMDB sentiment
classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from run_classifiers import generate_dirs
from data_helpers import load_data
import os



'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''


class CnnLstm(object):

    def __init__(self):
        self.filters = 64
        self.kernel_size = 5
        self.hidden_dims = 250
        self.pool_size = 4
        self.lstm_output_size = 70
        self.pretrained = True
        self.directory = "../results/models/"

    def run(self, maxlen, batch_size, epochs, verbose, data):

        x_train, x_test, y_train, y_test = data
        self.input_length = maxlen

        if verbose:
            print(len(x_train), 'train sequences')
            print(len(x_test), 'test sequences')

        if verbose:
            print('x_train shape:', x_train.shape)
            print('x_test shape:', x_test.shape)

            print('Build model...')

        model = Sequential()
        if self.pretrained:
            vocab_size = self.W.shape[0]
            embedding_dims = self.W.shape[1]
            model.add(Embedding(vocab_size, embedding_dims, weights=[self.W],
                                input_length=self.input_length))
        else:
            embedding_dims = 128

            model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
        model.add(Dropout(0.25))
        model.add(Conv1D(self.filters,
                         self.kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        model.add(LSTM(self.lstm_output_size))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        if verbose:
            print('Train...')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=verbose,
                  validation_data=(x_test, y_test))

        self.score, self.acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=verbose)

        if verbose:
            print('Test score:', self.score)
            print('Test accuracy:', self.acc)

        home = self.directory
        model.save(home + "cnn_lstm.h5")


if __name__ == '__main__':
    # Embedding
    max_features = 20000
    maxlen = 100
    embedding_dims = 128

    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4

    # LSTM
    lstm_output_size = 70

    # Training
    this_data_directory, this_results_directory = generate_dirs()
    batch_size = 32
    epochs = 4
    verbose = False
    raw_data = this_data_directory + "task1.train.txt"
    this_emb_file = this_data_directory + "GoogleNews-vectors-negative300.bin"
    this_emb_set = "w2v"
    data, avg_len, W, t = load_data(raw_data, this_emb_file, this_emb_set, True)
    cnn_lstm = CnnLstm()
    cnn_lstm.W = W
    cnn_lstm.pretrained = True
    lstm_results_dir = this_results_directory + this_emb_file + "/"
    if not os.path.exists(lstm_results_dir):
        os.makedirs(lstm_results_dir)
    cnn_lstm.directory = lstm_results_dir
    cnn_lstm.run(maxlen, batch_size, epochs, verbose, data)
