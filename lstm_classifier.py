'''Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from data_helpers import generate_dirs
from data_helpers import load_data
import os


class LstmModel(object):

    def __init__(self):
        self.lstm_output_size = 70

        self.pretrained = True
        self.W = []
        self.directory = "../results/models/"

    def run(self, max_len, batch_size, epochs, verbose, data):
        x_train, x_test, y_train, y_test = data
        self.input_length = max_len

        # print('Pad sequences (samples x time)')
        x_train = sequence.pad_sequences(x_train, maxlen=max_len)
        x_test = sequence.pad_sequences(x_test, maxlen=max_len)
        # print('x_train shape:', x_train.shape)
        # print('x_test shape:', x_test.shape)


        if verbose:
            print('Build model...')
        model = Sequential()
        if self.pretrained:
            vocab_size = self.W.shape[0]
            embedding_dims = self.W.shape[1]
            model.add(Embedding(vocab_size, embedding_dims, weights=[self.W],
                                input_length=self.input_length))
        else:
            embedding_dims = 128
            model.add(Embedding(max_len, embedding_dims))
        model.add(LSTM(embedding_dims, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        if verbose:
            print('Train...')
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

        home = self.directory
        model.save(home + "lstm.h5")

if __name__ == '__main__':

    data_directory, results_directory = generate_dirs()
    # Training
    batch_size = 32
    epochs = 4
    verbose = False
    raw_data = data_directory + "task1.train.txt"
    this_emb_file = data_directory + "GoogleNews-vectors-negative300.bin"
    this_emb_set = "w2v"
    data, avg_len, W, t = load_data(raw_data, this_emb_file, this_emb_set, True)
    lstm = LstmModel()
    lstm.W = W
    lstm.pretrained = True
    lstm_results_dir = results_directory + this_emb_file + "/"
    if not os.path.exists(lstm_results_dir):
        os.makedirs(lstm_results_dir)
    lstm.directory = lstm_results_dir
    lstm.run(avg_len, batch_size, epochs, verbose, data)
