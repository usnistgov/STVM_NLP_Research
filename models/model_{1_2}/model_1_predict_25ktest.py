import matplotlib
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
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


model = load_model('model_a_1.h5')

model.summary()

NUM_WORDS = 0
DISCRIMINATOR_CUTOFF = 5
SAMPLE_SIZE = 25000


# This function is reused from the model_1.py script; see that file for more information
def load_dataset_from_feat(directory, feat_file_name, use_for_predictions=False):
  data = {}
  data['reviews'] = []
  if not use_for_predictions:
    data['sentiments'] = []

  with open(os.path.join(directory, feat_file_name), 'r') as f:
    imdb_encoded_content = f.readlines()
    #if not use_for_predictions:
    # shuffle the reviews before using, only if training/testing but not if computing predictions for validation
    #random.shuffle(imdb_encoded_content)
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


# See note above the load_dataset_from_feat function definition
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


# See note above the load_dataset_from_feat function definition
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

print(test_data_full.shape)
print(test_data_full.head())

test_data = test_data_full[:][0:25000]

print(test_data.head())
print(test_data['reviews'].iloc[0])

test_data_mhe = weighted_multi_hot_sequences(test_data)
print(test_data_mhe.shape)

print(test_data_mhe)
omit_files = [".DS_Store"]


def lf():
    ff_pos_l = []
    ff_pos_r = []
    ff = []
    for filename in os.listdir('../test/pos/'):
        if filename not in omit_files:
            ff_pos_l.append(int(filename.split('_')[0]))
            ff_pos_r.append(filename.split('_')[1])
    #print(ff_pos_l[0], ff_pos_r[0])
    z = list(zip(ff_pos_l, ff_pos_r))
    z = sorted(z)
    ff_pos_l, ff_pos_r = zip(*z)
    for i in range(len(ff_pos_l)):
        ff.append(str(ff_pos_l[i]) + '_' + ff_pos_r[i])

    ff_neg_l = []
    ff_neg_r = []

    for filename in os.listdir('../test/neg/'):
        if filename not in omit_files:
            ff_neg_l.append(int(filename.split('_')[0]))
            ff_neg_r.append(filename.split('_')[1])
    z = list(zip(ff_neg_l, ff_neg_r))
    z = sorted(z)
    ff_neg_l, ff_neg_r = zip(*z)
    for i in range(len(ff_neg_l)):
        ff.append(str(ff_neg_l[i]) + '_' + ff_neg_r[i])
    return ff


list_of_names_of_test_files = lf()

# Print a few file names to verify that the data was split correctly
print(list_of_names_of_test_files[0])
print(list_of_names_of_test_files[1])

print(list_of_names_of_test_files[12500])
print(list_of_names_of_test_files[12501])
print(list_of_names_of_test_files[24999])
print(len(list_of_names_of_test_files))

# Evaluate the model on the multi-hot encodings
prediction_results = None
prediction_results = model.predict(test_data_mhe, verbose=1)

list_of_proba = []
list_of_file_name = []
print("predict shape", prediction_results.shape)

# Store a counter of the number of correct predictions over the (test) datset
correct_pred = 0
for i in range(SAMPLE_SIZE):
    proba = prediction_results[i][0]
#     Check if model prediction is correct and update counter accordingly
    if (proba < float(0.5) and i >= 12500) or (proba >= float(0.5) and i < 12500):
        correct_pred = correct_pred + 1
    list_of_proba.append(str(prediction_results[i][0]))
    list_of_file_name.append(list_of_names_of_test_files[i])

d_prob = {'prob': list_of_proba}
d_files = {'file': list_of_file_name}


dd = {'prob': list_of_proba, 'file': list_of_file_name}

# Save the model's predictions to a CSV file
df = pd.DataFrame(dd, index=None)
df.to_csv('model_a_25ktest.csv', index=False)

print("Accuracy on test: ", float(correct_pred)/float(25000))
