#!/usr/bin/python
# -*- coding: utf-8 -*-
import io, os
import numpy as np
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence



def generate_dirs():

    ######################################
    # Put your data in the data_directory
    # or update this path with your data
    # location.
    ######################################

    DATA_DIR = "../data/"
    RESULTS_DIR = "../results"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    data_directory = DATA_DIR
    results_directory = RESULTS_DIR

    return data_directory, results_directory


def tokenize(string):

    string = re.sub(r"[^A-Za-z0-9\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string.strip())

    return string

def load_split_docs(pos_file, neg_file):

    # Load data from files
    positive_examples = list(io.open(pos_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(io.open(neg_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = positive_examples + negative_examples
    t = Tokenizer()
    t.fit_on_texts(x_text)

    idxs = t.texts_to_sequences(x_text)
    idx_matrix = np.array(idxs)
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)


    x_train, x_test, y_train, y_test = train_test_split(
        idx_matrix, y, test_size=0.25, random_state=1234, shuffle=True)

    return x_train, x_test, y_train, y_test

def bigfile_reader(data_file, adjust):

    # Load data from file
    #docs = list(io.open(data_file, "r", encoding='utf-8').readlines())
    x_text = []
    original_y = []
    adjust_labels = adjust
    # for line in io.open(data_file, "r", encoding="utf-8"):
    for line in open(data_file, "r"):
        x_text.append(line.strip().split("\t")[0])
        if adjust_labels:
            original_y.append(line.strip().split("\t")[1])
        else:
            original_y.append(int(line.strip().split("\t")[1]))

    # x_text = [s.strip().split("\t")[0] for s in docs]

    # Generate labels

    if adjust_labels:
        y = adapt_labels(original_y)
    else:
         y = original_y

    # print(x_text, y)
    return x_text, y

def load_data(data_file, emb_file, emb_type, train):
    adjust_labels = True
    original_y, x_text = load_file(data_file, adjust_labels)

    if not train:
        adjust_labels = False

    # Generate labels
    if adjust_labels:
        y = adapt_labels(original_y)
    else:
        y = original_y

    avg_len, encoded,  t = vectorizer(x_text)

    if train:
        W = get_embeddings(emb_file, emb_type, t)
        # print("W shape: ", W.shape)
    else:
        W = []

    x_train, x_test, y_test, y_train = generate_splits(encoded,
                                                       t,
                                                       train,
                                                       x_text,
                                                       y)

    x_train = sequence.pad_sequences(x_train, maxlen=avg_len, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test, maxlen=avg_len, padding='post', truncating='post')

    data = x_train, x_test, y_train, y_test
    return data, avg_len, W, t


def new_load(x_file, dev_file, emb_file, emb_type, train):
    # train = True
    train_y, train_x = load_file(x_file, train)
    dev_y, dev_x = load_file(dev_file, train)
    dev_y= adapt_labels(dev_y)
    train_y= adapt_labels(train_y)

    avg_len, train_encoded,  t = vectorizer(train_x)
    dev_encoded = t.texts_to_sequences(dev_x)
    W = get_embeddings(emb_file, emb_type, t)

    x_train = sequence.pad_sequences(train_encoded, maxlen=avg_len, padding='post', truncating='post')
    x_dev = sequence.pad_sequences(dev_encoded, maxlen=avg_len, padding='post', truncating='post')

    data = x_train, x_dev, train_y, dev_y

    return data, avg_len, W, t



def vectorizer(x_text):
    t = Tokenizer()
    t.fit_on_texts(x_text)
    # Split by words
    encoded = t.texts_to_sequences(x_text)
    lengths = [len(x) for x in encoded]
    avg_len = int(np.mean(lengths))
    return avg_len, encoded, t


def load_file(data_file, adjust_labels):
    # Load data from file
    x_text = []
    original_y = []

    for line in open(data_file, "r"):
        x_text.append(line.strip().split("\t")[0])
        if adjust_labels:
            original_y.append(line.strip().split("\t")[-1])
        else:
            original_y.append(int(line.strip().split("\t")[-1]))
    return original_y, x_text


def generate_splits(encoded, t, train, x_text, y):
    if train:
        # split data into train and test
        x_train, x_test, y_train, y_test = train_test_split(
            encoded, y, test_size=0.25, random_state=1234, shuffle=True)
    else:

        x_train = t.texts_to_sequences(x_text)
        y_train = y
        x_test, y_test = [], []

    return x_train, x_test, y_test, y_train


def get_embeddings(emb_file, emb_type, t):
    if emb_type == "w2v":
        emb = load_bin_vec(emb_file, t.word_index)
    else:
        emb = load_glove(emb_file, t.word_index)
    print("num words found:" + str(len(emb)))
    add_unknown_words(emb, t.word_index, k=300)
    W, word_idx_map = get_W(emb, k=300)
    return W


def adapt_labels(array):

    y = np.array(array)
    y[y == "non-propaganda"] = 0
    y[y == "propaganda"] = 1

    return y


def word2index(dic, phrase):
    idx_list = []
    for word in phrase.split(" "):
        idx_list.append(dic[word])
    return idx_list


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        # ~ print header
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        # print(vocab_size)
        for line in range(vocab_size):
            # print(line)
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            # print(word)
            if word in vocab:
                # print(word)
                word_vecs[word] = np.frombuffer(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

    return word_vecs


def load_glove(fname, vocab):
    word_vecs = {}
    with open(fname, "rb") as f:
        for line in f:
            line = line.strip().split()
            word = line[0]
            if word in vocab:
                word_vecs[word] = np.array(line[1:], dtype='float32')

    return word_vecs

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)



if __name__ == '__main__':
    data_directory = generate_dirs()[0]

    train_file = data_directory + "task1.train.txt"
    y, x = load_file(train_file, True)
    x_train = x[:31900:]
    y_train = y[:31900:]
    train_x, dev_x, train_y, dev_y = train_test_split(x_train, y_train, test_size=0.095, random_state=1234, shuffle=True)
    print(len(dev_x))

