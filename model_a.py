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

from keras import backend as K
from keras.layers import BatchNormalization, InputLayer, RepeatVector
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Flatten
from keras.models import Sequential
import math
import random
import re
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('agg')


NUM_WORDS = 0
DISCRIMINATOR_CUTOFF = 5
L2_ETA = 0.019

# Load all data in a DataFrame.
# shuffles the data to ensure good mix of postive and negative reviews


def load_dataset_from_feat(directory, feat_file_name, use_for_predictions=False):
  data = {}
  data['reviews'] = []
  if not use_for_predictions:
    data['sentiments'] = []

  with open(os.path.join(directory, feat_file_name), 'r') as f:
    imdb_encoded_content = f.readlines()
    if not use_for_predictions:
      # shuffle the reviews before using, only if training/testing but not if computing predictions for validation
      random.shuffle(imdb_encoded_content)
    review_encoding = []
    for review in imdb_encoded_content:
      review_encoding = review.split()
      if not use_for_predictions:
        if int(review_encoding[0]) > DISCRIMINATOR_CUTOFF:
          data['sentiments'].append(1)
        else:
          data['sentiments'].append(0)
      review_encoding.pop(0)
      data['reviews'].append(review_encoding)
  return pd.DataFrame.from_dict(data)


def load_datasets_from_file():
    #  dataset = tf.keras.utils.get_file(
  #      fname='aclImdb.tar.gz',
  #      origin='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
  #      extract=True)
  # Assumes this script runs from the top directory containing the test and
  # train directory.
  global NUM_WORDS
  f = open("../imdb.vocab", "r")
  imdb_vocab = f.readlines()
  NUM_WORDS = len(imdb_vocab)
  print('Vocabulary size is: %d words' % (NUM_WORDS))
  '''
  with tf.io.gfile.GFile(os.path.join('..', 'imdb.vocab'), 'r') as f:
    imdb_vocab = f.readlines()
    NUM_WORDS = len(imdb_vocab)
    print('Vocabulary size is: %d words'%(NUM_WORDS))
    '''

  train_data = load_dataset_from_feat(
      os.path.join('..', 'train'), 'labeledBow.feat')
  test_data = load_dataset_from_feat(
      os.path.join('..', 'test'), 'labeledBow.feat')
  return train_data, test_data


def weighted_multi_hot_sequences(sequences):
    print("NUM_WORDS", NUM_WORDS)
    results = np.zeros((len(sequences['reviews']), NUM_WORDS))
    with open(os.path.join('..', 'imdbEr.txt'), 'r') as f:
        imdb_word_polarity = f.readlines()

    max = 0.0
    min = 0.0
    for review_index, review in enumerate(sequences['reviews']):
      for word in review:
        word_index, word_count = word.split(':')
        cumulative_polarity = int(word_count) * \
            float(imdb_word_polarity[int(word_index)])
        results[review_index, int(word_index)] = cumulative_polarity
        #accumulate statistics for the dataset
        if cumulative_polarity > max:
          max = cumulative_polarity
        elif cumulative_polarity < min:
          min = cumulative_polarity
    print('Dataset encoding stats: MIN = %f, MAX = %f\n' % (min, max))
    return results


print('Loading the large data set from disk...\n')
train_data_full, test_data_full = load_datasets_from_file()

print(train_data_full.shape)
print(train_data_full.head())

train_data = train_data_full[:][0:25000]
test_data = test_data_full[:][0:25000]

print(train_data.head())
print(train_data['reviews'].iloc[0])

train_data_mhe = weighted_multi_hot_sequences(train_data)
test_data_mhe = weighted_multi_hot_sequences(test_data)
print(train_data_mhe.shape)
print(test_data_mhe.shape)

print(train_data_mhe)


model = Sequential()

model.add(Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(
    L2_ETA), input_shape=(NUM_WORDS,)))
model.add(Dense(128, activation="relu",
                kernel_regularizer=keras.regularizers.l2(L2_ETA)))
model.add(Dense(64, activation="relu",
                kernel_regularizer=keras.regularizers.l2(L2_ETA)))


model.add(BatchNormalization())
model.add(RepeatVector(64))

model.add(LSTM(64, recurrent_dropout=0.3, return_sequences=True))
model.add(LSTM(64, recurrent_dropout=0.6, return_sequences=True))
model.add(LSTM(64, recurrent_dropout=0.3, return_sequences=True))

model.add(Flatten())


model.add(Dense(1, activation="sigmoid"))

#optimizer = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.summary()
#dice_coef_loss
#binary_crossentropy
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', 'binary_crossentropy'])


# checkpoint
filepath = "model_a.h5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(train_data_mhe, train_data['sentiments'], epochs=15, batch_size=512, validation_data=(
    test_data_mhe, test_data['sentiments']), shuffle=False, callbacks=callbacks_list, verbose=1)
