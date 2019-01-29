import json
import io
import numpy as np

def json2txt(infile, outfile):
    yelp_file = open(outfile, 'w')

    for line in open(infile, 'r'):
        elements = line.split('"')
        print(elements)
        yelp_file.write(elements[21].replace("\\n", "  ") + "\t" + elements[14].strip(",:") + "\t" + elements[3] + "\n")



def load_extracted_data(data_file):

    # Load data from file
    #docs = list(io.open(data_file, "r", encoding='utf-8').readlines())
    x_text = []
    original_y = []
    for line in io.open(data_file, "r", encoding='utf-8'):
        x_text.append(line.strip().split("\t")[0])
        original_y.append(int(line.strip().split("\t")[1]))

    # x_text = [s.strip().split("\t")[0] for s in docs]

    # Generate labels
    #
    y_ = np.array(original_y)
    y_[y_ <= 2] = 0
    y_[y_ >= 4] = 1

    y = y_
    print(y)


txtfile = './data/yelp_polarity.txt'
this_file = './data/yelp_onlytext.txt'
jsonfile = './data/yelp_academic_dataset_review.json'
load_extracted_data(this_file)
