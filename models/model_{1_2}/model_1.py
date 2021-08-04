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
# Use the AGG (Anti-Grain Geometry) backend since we are not displaying figures directly
matplotlib.use('agg')


NUM_WORDS = 0
DISCRIMINATOR_CUTOFF = 5
L2_ETA = 0.019

# Load all data in a DataFrame.
# shuffles the data to ensure good mix of postive and negative reviews


def sorted_files_list(directory):
    train_files_pos = []
    train_files_neg = []
    for filename in os.listdir(directory + "/pos/"):
        if filename != '.DS_Store':
              train_files_pos.append(filename)

    ll1 = list()
    ll2 = list()
    for f in train_files_pos:
        ss = f.split("_")
        ll1.append(int(ss[0]))
        ll2.append(ss[1])

    z = list(zip(ll1, ll2))
    z = sorted(z, key=lambda t: t[0])

    ll1, ll2 = zip(*z)

    train_files_pos = [str(ll1[i]) + "_" + ll2[i] for i in range(len(ll1))]
    print(train_files_pos[0], train_files_pos[1],
          train_files_pos[12498], train_files_pos[12499])

    for filename in os.listdir(directory + "/neg/"):
          if filename != '.DS_Store':
              train_files_neg.append(filename)

    ll1 = list()
    ll2 = list()
    for f in train_files_neg:
        ss = f.split("_")
        ll1.append(int(ss[0]))
        ll2.append(ss[1])

    z = list(zip(ll1, ll2))
    z = sorted(z, key=lambda t: t[0])

    ll1, ll2 = zip(*z)

    train_files_neg = [str(ll1[i]) + "_" + ll2[i] for i in range(len(ll1))]
    print(train_files_neg[0], train_files_neg[1],
          train_files_neg[12498], train_files_neg[12499])

    return train_files_pos, train_files_neg

# Loads a feature file and returns a dictionary of its data
# Takes as arguments:
# - strings representing the directory and file name of the feature file (bag-of-words representations of the reviews)
# - a boolean indicating whether to include the sentiment values in the returned dictionary (a value of False *will* include them)
def load_dataset_from_feat(directory, feat_file_name, use_for_predictions=False):
  data = {}
  data['reviews'] = []

  print(os.path.join(directory, feat_file_name))

  if not use_for_predictions:
    data['sentiments'] = []

#   Open the feature file and load its lines (each review is on a separate line)
  with open(os.path.join(directory, feat_file_name), 'r') as f:
    imdb_encoded_content = f.readlines()
    #if not use_for_predictions:
    # shuffle the reviews before using, only if training/testing but not if computing predictions for validation
    #random.shuffle(imdb_encoded_content)
    print("************************************")
    #print(imdb_encoded_content)
    print("************************************")
    review_encoding = []
#     Loop through bag-of-words encoding of each review's vocabulary
    for review in imdb_encoded_content:
#       Split on whitespace (e.g., "0:6 1:11 2:4 3:5 ..." becomes ["0:6", "1:11", "2:4", "3:5", ...])
      review_encoding = review.split()
      if not use_for_predictions:
        if int(review_encoding[0]) > DISCRIMINATOR_CUTOFF:
          data['sentiments'].append(1)
        else:
          data['sentiments'].append(0)
#       Remove label from the list of frequencies (first element in each line/review)
      review_encoding.pop(0)
      data['reviews'].append(review_encoding)
  return pd.DataFrame.from_dict(data)


# Loads the dataset and prints information about the vocabulary
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
  

  train_data = load_dataset_from_feat(
      os.path.join('..', 'train'), 'labeledBow.feat')
  #test_data = load_dataset_from_feat(
  #    os.path.join('..', 'test'), 'labeledBow.feat')
  return train_data  # , test_data

# Given a dictionary containing a list of reviews from the dataset (with the key 'reviews'):
# Generate a matrix (dense NumPy array) of multi-hot encoded vectors for each review in the dataset
# Resulting matrix dimensions are [number of reviews] by [number of words in vocabulary]
# Note that this typically creates a very large array (several gigabytes) and the script may crash if not enough memory is available to allocate
def weighted_multi_hot_sequences(sequences):
    print("NUM_WORDS", NUM_WORDS)
#     Create base matrix
    results = np.zeros((len(sequences['reviews']), NUM_WORDS))
#     Load vocabulary word polarity values
    with open(os.path.join('..', 'imdbEr.txt'), 'r') as f:
        imdb_word_polarity = f.readlines()

#     Initialize variables to store summary statistics
    max = 0.0
    min = 0.0
#     Loop through each review -> each word in the review (i.e., each unique word in the bag-of-words data)
    for review_index, review in enumerate(sequences['reviews']):
      for word in review:
        word_index, word_count = word.split(':')
#         Compute a cumulative polarity score that combines the pre-generated word polarities with their frequencies in a given review
        cumulative_polarity = int(word_count) * \
            float(imdb_word_polarity[int(word_index)])
#         Set the value in the matrix
        results[review_index, int(word_index)] = cumulative_polarity
        # Accumulate statistics for the dataset
        if cumulative_polarity > max:
          max = cumulative_polarity
        elif cumulative_polarity < min:
          min = cumulative_polarity
#     Display the lowest and highest cumulative polarity scores
    print('Dataset encoding stats: MIN = %f, MAX = %f\n' % (min, max))
    return results


print('Loading the large data set from disk...\n')
#train_data_full, test_data_full = load_datasets_from_file()
train_data_full = load_datasets_from_file()

train_files_list = []
train_files_pos, train_files_neg = sorted_files_list("../train")
train_files_list = train_files_pos + train_files_neg

print(train_data_full.shape)
print(train_data_full.head())
print(train_files_list[0])
print(train_files_list[1])


# Get the first 25000 samples from the loaded dataset
train_data = train_data_full[:][0:25000]

print(train_data.head())
print(train_data['reviews'].iloc[0])

# Create multi-hot encoded array for the training data samples
# (note that this will be a 2D array with axis 0 being the reviews/batch dimension and axis 1 being the encoded vocabulary (combined frequency/polarity scores))
train_data_mhe = weighted_multi_hot_sequences(train_data)
print(train_data_mhe.shape)

print(train_data_mhe[0])
print(len(train_data_mhe[0]))

# Number of training and validation samples
TRAINING_SAMPLE = 20000
VALIDATION_SAMPLE = 5000
df_train = pd.read_csv('imdb_train_20k.csv')
df_validation = pd.read_csv('imdb_train_5k.csv')
SAMPLE_SIZE_TRD = len(df_train)
SAMPLE_SIZE_VLD = len(df_validation)
# Preview the training data
print(df_train.head())

# Ensures the number of samples in both of the csv files (tables) match the expected numbers
assert SAMPLE_SIZE_TRD == TRAINING_SAMPLE, 'training sample not complete....'
assert SAMPLE_SIZE_VLD == VALIDATION_SAMPLE, 'validation sample not complete....'

# Shuffle the training data DataFrame
df_train = df_train.sample(frac=1)

# Create the actual arrays that the model will be trained on
# [train/validation]_x are the multi-hot encodings (floats), [train/validation]_y are the binary sentiment classifications (ints)
train_y = np.zeros([TRAINING_SAMPLE, 1], dtype=np.int)
train_x = np.zeros([TRAINING_SAMPLE, 89527], dtype=np.float64)
validation_y = np.zeros([VALIDATION_SAMPLE, 1], dtype=np.int)
validation_x = np.zeros([VALIDATION_SAMPLE, 89527], dtype=np.float64)

for index in df_train.index:
    file_name = str(df_train['file'][index])
    label = int(df_train['label'][index])

    index_in_files_list = train_files_list.index(file_name)
    train_x[index] = train_data_mhe[index_in_files_list]
    train_y[index] = label

print(train_x[0])
print(train_y[0])

for index in df_validation.index:
    file_name = str(df_validation['file'][index])
    label = int(df_validation['label'][index])

    index_in_files_list = train_files_list.index(file_name)
    validation_x[index] = train_data_mhe[index_in_files_list]
    validation_y[index] = label

model = Sequential()

model.add(Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(
    L2_ETA), input_shape=(NUM_WORDS,)))
model.add(Dropout(0.6))
model.add(Dense(128, activation="relu",
                kernel_regularizer=keras.regularizers.l2(L2_ETA)))
model.add(Dropout(0.4))
model.add(Dense(64, activation="relu",
                kernel_regularizer=keras.regularizers.l2(L2_ETA)))

model.add(BatchNormalization())
model.add(RepeatVector(96))

model.add(LSTM(64, recurrent_dropout=0.4, return_sequences=True)) 
model.add(LSTM(64, recurrent_dropout=0.7, return_sequences=True))
model.add(LSTM(64, recurrent_dropout=0.4, return_sequences=True))

model.add(Flatten())


model.add(Dense(1, activation="sigmoid"))

#optimizer = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy', 'binary_crossentropy'])


# checkpoint
filepath = "model_1.h5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(train_x, train_y, validation_data=(validation_x, validation_y),
          epochs=150, batch_size=512,
          shuffle=False, verbose=1, callbacks=callbacks_list)
model.save(filepath)
