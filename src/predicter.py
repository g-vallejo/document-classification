from keras.models import load_model
from data_helpers import load_data
from keras.preprocessing import sequence
from data_helpers import generate_dirs
import os


def predict(data, model_file):

    model = load_model(model_file)

    predictions = model.predict_classes(data)
    return predictions


def save_propaganda_format(predictions, ids, name):
    with open(name, "w+") as foutput:
        for i in range(len(predictions)):

            if predictions[i]:
                prediction = 'propaganda'
            else:
                prediction = 'non-propaganda'

            foutput.write("%s\t%s\n" % (ids[i], prediction))

def load_test(infile):
    text = []
    ids = []
    for line in open(infile, "r"):
        text.append(line.strip().split("\t")[0])
        ids.append(line.strip().split("\t")[1])
    return text, ids


data_directory, results_directory = generate_dirs()

this_emb_file = data_directory + "GoogleNews-vectors-negative300.bin"
this_emb_set = "w2v"
this_train = data_directory + "task1.train.txt"
this_dev = data_directory + "task1.dev.txt"
this_test = data_directory + "task1.test.txt"
this_data, this_input_length, this_W, t = load_data(this_train,
                                                    this_emb_file,
                                                    this_emb_set,
                                                    True)
test_set, this_ids = load_test(this_test)
encoded_test = t.texts_to_sequences(test_set)
padded_test = sequence.pad_sequences(encoded_test, maxlen=this_input_length, padding='post', truncating='post')

embs = ["urban", "levy", "fasttext", "komninos"]
this_directory = results_directory + "/models/"
if not os.path.exists(this_directory):
    os.makedirs(this_directory)
models = ["cnn"]
dev_write_directory = "results_dev/"
test_write_directory = "results_test/"
for name in embs:
    for model in models:
        this_modelf = this_directory + name + "/" + model + ".h5"
        this_predictions = predict(padded_test, this_modelf)
        saved_file = this_directory + dev_write_directory + name + model
        save_propaganda_format(this_predictions, this_ids, saved_file)
