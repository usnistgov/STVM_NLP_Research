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

import tensorflow as tf
from collections import defaultdict
import numpy as np
import requests
import zipfile
import random
import os
import re

import matplotlib.pyplot as plt

SAMPLE_SIZE = 25000
BASE_DIR = '../'


class RawDataset:
    def __init__(self, path):
        self.train_dir = path + "train/"
        self.test_dir = path + "test/"
        self.sample = ['pos/', 'neg/']

    def get_dataset(self, dir_path, shuffle=False):
        _list = []
        for sample in self.sample:
            for file in os.listdir(dir_path + sample):
                _list.append(dir_path + sample + file)

        assert len(
            _list) == SAMPLE_SIZE, "Dataset not equal to SAMPLE_SIZE, aborting ...."

        if shuffle:
            random.shuffle(_list)
        return _list

    def get_train_dataset(self):
        return self.get_dataset(dir_path=self.train_dir, shuffle=True)

    def get_test_dataset(self):
        return self.get_dataset(dir_path=self.test_dir, shuffle=True)


class Embedding:
    def __init__(self, embedding_size, max_length):
        self.glove_zipfile = 'glove.6B.zip'
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.embeddings = [50, 100, 200, 300]
        self.url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        self.downloader()

    def downloader(self):
        if not os.path.exists(self.glove_zipfile):
            print("downloading glove embedding .....")
            r = requests.get(self.url, self.glove_zipfile)
            if r.status_code != 200:
                print('unable to download file')
        glove_embedding_file = 'glove.6B.{}d.txt'.format(self.embedding_size)
        if not os.path.exists(glove_embedding_file) and self.embedding_size in self.embeddings:
            with zipfile.ZipFile(self.glove_zipfile, 'r') as glove_zip:
                glove_zip.extractall()

    def get_embedding(self):
        glove_embedding_file = 'glove.6B.{}d.txt'.format(self.embedding_size)
        with open(glove_embedding_file, 'r') as f:
            words = defaultdict(int)
            vectors = defaultdict(lambda: np.zeros([self.embedding_size]))

            index = 1
            for line in f:
                fields = line.split()
                word = str(fields[0])
                vector = np.asarray(fields[1:], np.float32)
                words[word] = index
                vectors[index] = vector
                index += 1
        return words, vectors

    def sanitize(self, list_of_files, list_of_reviews, y):
        for index in range(len(list_of_files)):
            file_path = list_of_files[index]
            with open(file_path, 'r') as f:
                review = f.read()
                review = review.lower()
                review = review.replace("<br />", " ")
                review = re.sub(r"[^a-z ]", " ", review)
                review = re.sub(r" +", " ", review)
                review = review.split(" ")
                list_of_reviews.append(review)
            if file_path.split("/")[2] == "pos":
                y[index] = 1

    def build_dataset(self, list_of_files):
        words, vectors = self.get_embedding()
        list_of_reviews = []
        padded_data_list = []
        y = np.zeros([SAMPLE_SIZE, 1], dtype=np.int)
        self.sanitize(list_of_files, list_of_reviews, y)
        for review in list_of_reviews:
            rv = [words[word] for word in review]
            padded_rv = [0] * self.max_length
            a = min(len(rv), self.max_length)
            padded_rv[:a] = rv[:a]
            padded_data_list.append(padded_rv)

        data_list = []
        for padded_reviews in padded_data_list:
            data_list.append([vectors[word] for word in padded_reviews])

        return np.array(data_list), y


def main():
    ES = 100
    MXLEN = 500
    r_ds = RawDataset(path=BASE_DIR)
    list_of_train_files = r_ds.get_train_dataset()
    list_of_test_files = r_ds.get_test_dataset()

    embedding = Embedding(ES, MXLEN)
    x_train, y_train = embedding.build_dataset(list_of_train_files)
    x_test, y_test = embedding.build_dataset(list_of_test_files)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    assert x_train.shape == x_test.shape, 'train_x and test_x dataset are not of same shape'
    assert y_train.shape == y_test.shape, 'train_y and test_y dataset are not of same shape'

    BLSTM_UNITS = 64

    input1 = tf.keras.layers.Input(shape=(MXLEN, ES))
    model_Blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        BLSTM_UNITS, dropout=0.2, recurrent_dropout=0.4, return_sequences=True))(input1)
    model_Blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        BLSTM_UNITS, dropout=0.2, recurrent_dropout=0.4, return_sequences=True))(model_Blstm)

    model_attention = tf.keras.layers.Dense(1, activation="tanh")(model_Blstm)
    model_attention = tf.keras.layers.Flatten()(model_attention)
    model_attention = tf.keras.layers.Activation("softmax")(model_attention)
    model_attention = tf.keras.layers.RepeatVector(
        BLSTM_UNITS * 2)(model_attention)
    model_attention = tf.keras.layers.Permute([2, 1])(model_attention)

    latent_representation = tf.keras.layers.multiply(
        [model_Blstm, model_attention])
    latent_representation = tf.keras.layers.Lambda(lambda xin: tf.keras.backend.sum(
        xin, axis=-2), output_shape=(BLSTM_UNITS * 2,))(latent_representation)

    output_layer = tf.keras.layers.Dense(
        1, activation='sigmoid')(latent_representation)

    model = tf.keras.Model(inputs=input1, outputs=output_layer)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_crossentropy'])

    filepath = "model_b.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = model.fit(x_train, y_train, epochs=15, batch_size=256, validation_data=(
        x_test, y_test), shuffle=False, callbacks=callbacks_list, verbose=1)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.show()


if __name__ == '__main__':
    main()
