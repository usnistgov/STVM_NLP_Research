import numpy as np
from keras.utils import plot_model
from keras.layers import merge
from keras.layers import BatchNormalization, InputLayer, RepeatVector, Permute
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Flatten, Bidirectional
from keras.models import Sequential, Model
import tensorflow as tf
#from tensorflow import keras
import keras.backend as K

import keras
import os
import pandas as pd
import random
import os
import pandas as pd
from collections import defaultdict
import zipfile
import requests
import re


TRAINING_SAMPLE = 20000
VALIDATION_SAMPLE = 5000
EMBEDDING_SIZE = 100
MAX_REVIEW_LEN = 800
BLSTM_UNITS = 64

train_csv_file = 'imdb_train_20k.csv'
val_csv_file = "imdb_train_5k.csv"

train_y = np.zeros([TRAINING_SAMPLE, 1], dtype=np.int)
val_y = np.zeros([VALIDATION_SAMPLE, 1], dtype=np.int)

list_of_train_reviews = list()
list_of_validation_reviews = list()

glove_file = "glove.6B.zip"


# Download GloVe word embeddings and extract from the zip file if necessary
def download(url_):
    if not os.path.exists(glove_file):
        print("downloading glove embedding .....")
        r = requests.get(url_, glove_file)
    glove_filename = "glove.6B.{}d.txt".format(EMBEDDING_SIZE)
    if not os.path.exists(glove_filename) and EMBEDDING_SIZE in [50, 100, 200, 300]:
        print("extract glove embeddings ...")
        with zipfile.ZipFile(glove_file, 'r') as z:
            z.extractall()


# Load word embeddings and indices (returns 2 arrays)
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


#list_of_filename = list()
list_of_y = list()

#list_of_filename_val = list()
list_of_y_val = list()

# Process data and generate training sample
def create_training_sample(filename, if_train):
#     Load samples from CSV file
    df_train = pd.read_csv(filename)
    SAMPLE_SIZE = len(df_train)
#     Check that number of samples in dataset (i.e., DataFrame loaded from `filename`) is correct
    if if_train:
        assert SAMPLE_SIZE == TRAINING_SAMPLE, 'training sample not complete....'
    else:
        assert SAMPLE_SIZE == VALIDATION_SAMPLE, 'validation sample not complete....'

    if if_train:
#         Shuffle training data
        df_train = df_train.sample(frac=1)

#     Loop through reviews in dataset
    for index in df_train.index:
        review = str(df_train['review'][index])
        label = int(df_train['label'][index])
        #list_of_filename.append(df_train['file'][index])
        review = review.lower()
        
#         Remove newline tags
        review = review.replace("<br />", " ")
#         Replace characters other than letters and spaces with whitespace
        review = re.sub(r"[^a-z ]", " ", review)
#         Remove consecutive spaces in review
        review = re.sub(r" +", " ", review)

#         Split review string into words
        review = review.split(" ")

#         Add the review and label to the relevant lists
        if if_train:
            list_of_train_reviews.append(review)
            list_of_y.append(label)
        else:
            list_of_validation_reviews.append(review)
            list_of_y_val.append(label)


# Get index of each review in dataset
def encode_reviews(revs):
    train_data = []
    for review in revs:
        int_review = [word_to_int[word] for word in review]
        train_data.append(int_review)
    return train_data


create_training_sample(filename=train_csv_file, if_train=True)
create_training_sample(filename=val_csv_file, if_train=False)

print(list_of_train_reviews[0])
print(list_of_validation_reviews[0])

for index in range(len(list_of_y)):
    train_y[index] = list_of_y[index]

for index in range(len(list_of_y_val)):
    val_y[index] = list_of_y_val[index]

train_reviews = encode_reviews(list_of_train_reviews)
val_reviews = encode_reviews(list_of_validation_reviews)
print(train_reviews[0])
print(val_reviews[0])

# Pad each sample to the same length
def zero_pad_reviews(revs):
    _data_padded = []
    for review in revs:
        padded = [0] * MAX_REVIEW_LEN
        stop_index = min(len(review), MAX_REVIEW_LEN)
        padded[:stop_index] = review[:stop_index]
        _data_padded.append(padded)
    return _data_padded


train_reviews = zero_pad_reviews(train_reviews)
val_reviews = zero_pad_reviews(val_reviews)

# Get the word embeddings corresponding to the content of each review
def review_ints_to_vecs(train_reviews):
    train_data = []
    for review in train_reviews:
        vec_review = [int_to_vec[word] for word in review]
        train_data.append(vec_review)
    return train_data


train_reviews = np.array(review_ints_to_vecs(train_reviews))
val_reviews = np.array(review_ints_to_vecs(val_reviews))

print(train_reviews.shape, train_y.shape)
print(val_reviews.shape, val_y.shape)


input1 = Input(shape=(MAX_REVIEW_LEN, EMBEDDING_SIZE))
model_lstm = Bidirectional(LSTM(
    BLSTM_UNITS, dropout=0.2, recurrent_dropout=0.4, return_sequences=True))(input1)

model_lstmNB = Bidirectional(LSTM(
    BLSTM_UNITS, dropout=0.2, recurrent_dropout=0.4, return_sequences=True))(model_lstm)

model_attention = Dense(1, activation="tanh")(model_lstm)
model_attention = Flatten()(model_attention)
model_attention = Activation("softmax")(model_attention)
model_attention = RepeatVector(BLSTM_UNITS*2)(model_attention)
model_attention = Permute([2, 1])(model_attention)

latent_representation = keras.layers.multiply([model_lstmNB, model_attention])
latent_representation = keras.layers.Lambda(lambda xin: K.sum(
    xin, axis=-2), output_shape=(BLSTM_UNITS*2, ))(latent_representation)

output_layer = Dense(1, activation='sigmoid')(latent_representation)

model = Model(inputs=input1, outputs=output_layer)

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', 'binary_crossentropy'])


filepath = "model_2.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


model.fit(train_reviews, train_y, validation_data=(val_reviews, val_y),
            epochs=200, batch_size=512, verbose=1, callbacks=callbacks_list)
model.save(filepath)

