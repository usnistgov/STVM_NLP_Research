#import matplotlib
from tensorflow import keras
import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import defaultdict
import re
import zipfile
import requests
import random
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers import BatchNormalization, InputLayer, RepeatVector
from keras.models import load_model


model = load_model('model_b_2.h5')

model.summary()


SAMPLE_SIZE = 25000

omit_files = [".DS_Store"]


# Get a list of the file names of the text files containing the reviews in the testing data
def list_of_test_files():
    sample_dir_list = ['../test/pos/', '../test/neg/']
    list_of_names_of_test_files = []
#     Loop through sub-directories in sample_dir_list
    for sample_list_index in range(len(sample_dir_list)):
        for file in os.listdir(sample_dir_list[sample_list_index]):
#             Add the (relative) file path to the list
            if file not in omit_files:
                list_of_names_of_test_files.append(
                    sample_dir_list[sample_list_index] + file)
    return list_of_names_of_test_files


list_of_names_of_test_files = list_of_test_files()
#print(list_of_names_of_train_files)
print('list_of_names_of_test_files length: ',
      len(list_of_names_of_test_files))

assert len(list_of_names_of_test_files) == SAMPLE_SIZE

#random.shuffle(list_of_names_of_train_files)


glove_file = "glove.6B.zip"
EMBEDDING_SIZE = 100


# This function is reused from model_2.py; refer to that script for more information
def download(url_):
    if not os.path.exists(glove_file):
        print("downloading glove embedding .....")
        r = requests.get(url_, glove_file)
    glove_filename = "glove.6B.{}d.txt".format(EMBEDDING_SIZE)
    if not os.path.exists(glove_filename) and EMBEDDING_SIZE in [50, 100, 200, 300]:
        print("extract glove embeddings ...")
        with zipfile.ZipFile(glove_file, 'r') as z:
            z.extractall()


# See the note above the download function
def load_glove():
    with open("glove.6B.100d.txt", 'r') as glove_vectors:
        word_to_int = defaultdict(int)
        int_to_vec = defaultdict(lambda: np.zeros([EMBEDDING_SIZE]))

        index = 1
        for line in glove_vectors:
            fields = line.split()
            word = str(fields[0])
            vec = np.asarray(fields[1:], np.float32)
            word_to_int[word] = index
            int_to_vec[index] = vec
            index += 1
    return word_to_int, int_to_vec


download("http://nlp.stanford.edu/data/glove.6B.zip")
word_to_int, int_to_vec = load_glove()

list_of_test_reviews = []


def get_sanitized_reviews(list_of_file_paths, of_type):
    for i in range(len(list_of_file_paths)):
        file_path = list_of_file_paths[i]
        with open(file_path, 'r') as f:
            review = f.read()
            review = review.lower()
            review = review.replace("<br />", " ")
            review = re.sub(r"[^a-z ]", " ", review)
            review = re.sub(r" +", " ", review)
            review = review.split(" ")
            list_of_test_reviews.append(review)


# See the note above the download function
def encode_reviews(revs):
    test_data = []
    for review in revs:
        int_review = [word_to_int[word] for word in review]
        test_data.append(int_review)
    return test_data


get_sanitized_reviews(list_of_names_of_test_files, "test")
print(list_of_test_reviews[0])

predict_reviews = encode_reviews(list_of_test_reviews)
print(predict_reviews[0])
print('list_of_names_of_test_files: ', list_of_names_of_test_files[0])

MAX_REVIEW_LEN = 1200


# See the note above the download function
def zero_pad_reviews(revs):
    _data_padded = []
    for review in revs:
        padded = [0] * MAX_REVIEW_LEN
        stop_index = min(len(review), MAX_REVIEW_LEN)
        padded[:stop_index] = review[:stop_index]
        _data_padded.append(padded)
    return _data_padded


predict_reviews = zero_pad_reviews(predict_reviews)


# See the note above the download function
def review_ints_to_vecs(train_reviews):
    train_data = []
    for review in train_reviews:
        vec_review = [int_to_vec[word] for word in review]
        train_data.append(vec_review)
    return train_data


predict_reviews = np.array(review_ints_to_vecs(predict_reviews))
print(predict_reviews.shape)

prediction_results = None
prediction_results = model.predict(predict_reviews, verbose=1)
#prediction_results = model.predict_proba(predict_reviews, verbose=1)
#print(prediction_results)
#print(prediction_results[0][0])

list_of_proba = []
list_of_file_name = []
print("predict shape", prediction_results.shape)
correct_pred = 0
for i in range(SAMPLE_SIZE):
    proba = prediction_results[i][0]
    if (proba < float(0.5) and 'neg' in list_of_names_of_test_files[i]) or (proba >= float(0.5) and 'pos' in list_of_names_of_test_files[i]):
        correct_pred = correct_pred + 1

    list_of_proba.append(str(prediction_results[i][0]))
    list_of_file_name.append(list_of_names_of_test_files[i].split('/')[-1])

print("Accuracy: ", float(correct_pred)/float(25000))

d_prob = {'prob': list_of_proba}
d_files = {'file': list_of_file_name}

dd = {'prob': list_of_proba, 'file': list_of_file_name}

df = pd.DataFrame(dd, index=None)
df.to_csv('model_b_2_25ktest.csv', index=False)

