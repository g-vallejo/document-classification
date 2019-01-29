from lstm_classifier import LstmModel
from cnn_classifier import CNN as cnn
from cnn_lstm_classifier import CnnLstm
from data_helpers import load_data, new_load, generate_dirs
import os





def run_all_models(parameters):
    input_length, batch_size, W, epochs, verbose, data, emb_set, results_dir = parameters
    results = {}
    models_directory = results_dir + "models/"
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)
    print len(data[0]), 'train sequences'
    print len(data[1]), 'test sequences'
    classifiers = [cnn(), LstmModel(), CnnLstm()]
    specific_model_directory = results_dir + emb_set + "/"
    if not os.path.exists(specific_model_directory):
        os.makedirs(specific_model_directory)
    for clf in classifiers:
        clf_name = str(clf.__str__()).split(".")[0].strip("<")
        clf.W = W
        clf.pretrained = True
        clf.directory = specific_model_directory
        clf.run(input_length, batch_size, epochs, verbose, data)
        results[clf_name] = {"acc": clf.acc, "loss": clf.score}

    for approach, scores in results.items():
        print (approach + ": " + str(scores.items()))


def run_several_datasets(data_list, emb_file, emb):
    for dataset in data_list:
        data, input_length, max_len, W, t = load_data(dataset, emb_file, emb, True)
        print("\n==============================\n" + dataset + "\n==============================\n")
        parameters = input_length, max_len, batch_size, W, epochs, verbose, data
        run_all_models(parameters)

def run_several_embs(data_train, data_dev, emb_file_dic):

    for emb_set in emb_file_dic.keys():
        emb_file = emb_file_dic[emb_set]
        print("\n==============================" + emb_set.upper() + "==============================\n")
        if type(data_train) == list:
            run_several_datasets(data_train, emb_file, emb_set)
        else:
            data, input_length, W, t = new_load(data_train, data_dev, emb_file, emb_set, True)
            parameters = input_length, batch_size, W, epochs, verbose, data, emb_set
            run_all_models(parameters)


if __name__ == '__main__':

    batch_size = 32
    epochs = 5
    verbose = True
    directories = generate_dirs()
    this_data_directory, this_results_directory = directories
    raw_train_data = this_data_directory + "task1.train.txt"
    raw_dev_data = this_data_directory + "task1.dev.txt"
    emb_dict ={"w2v": "GoogleNews-vectors-negative300.bin",
               "fasttext": "wiki-news-300d-1M.vec",
               "glove": "glove.6B.300d.txt",
               "urban": "urban_glove200_lower.txt",
               "komninos": "komninos_english_embeddings",
               "levy": "levy_english_dependency_embeddings"}
    run_several_embs(raw_train_data, raw_dev_data, emb_dict)
