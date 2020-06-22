'''
==================================================LICENSING TERMS==================================================
This code and data was developed by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States and are considered to be in the public domain. The code and data is provided by NIST as a public service and is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR STATUTORY, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST does not warrant or make any representations regarding the use of the data or the results thereof, including but not limited to the correctness, accuracy, reliability or usefulness of the data. NIST SHALL NOT BE LIABLE AND YOU HEREBY RELEASE NIST FROM LIABILITY FOR ANY INDIRECT, CONSEQUENTIAL, SPECIAL, OR INCIDENTAL DAMAGES (INCLUDING DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, AND THE LIKE), WHETHER ARISING IN TORT, CONTRACT, OR OTHERWISE, ARISING FROM OR RELATING TO THE DATA (OR THE USE OF OR INABILITY TO USE THIS DATA), EVEN IF NIST HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

To the extent that NIST may hold copyright in countries other than the United States, you are hereby granted the non-exclusive irrevocable and unconditional right to print, publish, prepare derivative works and distribute the NIST data, in any medium, or authorize others to do so on your behalf, on a royalty-free basis throughout the world.

You may improve, modify, and create derivative works of the code or the data or any portion of the code or the data, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the code or the data and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the code or the data: Citation recommendations are provided below. Permission to use this code and data is contingent upon your acceptance of the terms of this agreement and upon your providing appropriate acknowledgments of NIST's creation of the code and data.

Paper Title:
    SSNet: a Sagittal Stratum-inspired Neural Network Framework for Sentiment Analysis

SSNet authors and developers:
    Apostol Vassilev:
        Affiliation: National Institute of Standards and Technology
        Email: apostol.vassilev@nist.gov
    Munawar Hasan:
        Affiliation: National Institute of Standards and Technology
        Email: munawar.hasan@nist.gov
====================================================================================================================
'''

from keras.datasets import imdb
import matplotlib
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import threading
from collections import defaultdict
import re
import random
import math
from keras.models import load_model

PROBABILITY_DISCRIMINATOR_CUTOFF = 0.5

model = load_model("model_b.h5")
model.summary()


(x_train, y_train), (x_test, y_test) = imdb.load_data()


word_index = imdb.get_word_index()
index = dict([(value, key) for (key, value) in word_index.items()])


comments = []
list_of_train_comments = []
list_of_test_comments = []
predict_y = np.zeros([50000, 1])


def create_samples(of_type):
    print("creating samples ..." + str(of_type))
    for sample_index in range(25000):
        sample = None
        list_token = []
        if of_type == "train":
            sample = x_train[sample_index]
        else:
            sample = x_test[sample_index]

        for i in range(len(sample)):
            keras_numeral = sample[i]
            if str(index.get(keras_numeral - 3)) == "None":
                continue
            else:
                list_token.append(str(index.get(keras_numeral - 3)))
        if of_type == "train":
            list_of_train_comments.append(list_token)
        else:
            list_of_test_comments.append(list_token)


list_of_tasks = ["train", "test"]
list_of_threads = []
for item in list_of_tasks:
    t = threading.Thread(target=create_samples, args=(item,))
    list_of_threads.append(t)
    t.start()
for t in list_of_threads:
    t.join()

comments = list_of_train_comments + list_of_test_comments


EMBEDDING_SIZE = 100
MAX_REVIEW_LEN = 500


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


word_to_int, int_to_vec = load_glove()


def encode_reviews(revs):
    train_data = []
    for review in revs:
        int_review = [word_to_int[word] for word in review]
        train_data.append(int_review)
    return train_data


predict_reviews = encode_reviews(comments)


def zero_pad_reviews(revs):
    _data_padded = []
    for review in revs:
        padded = [0] * MAX_REVIEW_LEN
        stop_index = min(len(review), MAX_REVIEW_LEN)
        padded[:stop_index] = review[:stop_index]
        _data_padded.append(padded)
    return _data_padded


predict_reviews = zero_pad_reviews(predict_reviews)


def review_ints_to_vecs(train_reviews):
    train_data = []
    for review in train_reviews:
        vec_review = [int_to_vec[word] for word in review]
        train_data.append(vec_review)
    return train_data


predict_reviews = np.array(review_ints_to_vecs(predict_reviews))


prediction_results = None
prediction_results = model.predict(predict_reviews, verbose=1)
print(prediction_results)


file_name = "model_b.txt"
s = ""
for i in range(0, len(prediction_results)):
    s = s + str(format(float(prediction_results[i][0]), ".8f"))
    if i != len(prediction_results) - 1:
        s = s + "\n"

f = open(file_name, "w")
f.write(s)
f.close()
