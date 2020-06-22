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

import matplotlib
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import random
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers import BatchNormalization, InputLayer, RepeatVector
from keras.models import load_model


model = load_model('model_a.h5')

model.summary()


NUM_WORDS = 89527
DISCRIMINATOR_CUTOFF = 5
L2_ETA = 0.019
PROBABILITY_DISCRIMINATOR_CUTOFF = 0.5


def computePredictionsScore(sentiments_file_name, labels_file_name):
  accurate_verdicts = 0
  accurate_positive_verdicts = 0
  accurate_negative_verdicts = 0
  total_verdicts = 0
  total_positive_verdicts = 0
  total_negative_verdicts = 0
# compute separate postive, negative and total scores
  print("prepare to load the data from the test...\n")
  with open(os.path.join('.', sentiments_file_name), "r") as f:
    computed_sentiments = f.readlines()
    total_verdicts = len(computed_sentiments)
    with tf.io.gfile.GFile(os.path.join('..', labels_file_name), "r") as f_in:
      labeled_input = f_in.readlines()
      for sentiment_index, sentiment in enumerate(computed_sentiments):
        parsed_sentiments = sentiment.split()
        verdict = parsed_sentiments[1]
        if verdict == 'neg':
          total_negative_verdicts += 1
        else:
          total_positive_verdicts += 1
        review = labeled_input[sentiment_index].split()
        label = int(review[0])
        if label == 0 and verdict == 'neg':
          accurate_verdicts += 1
          accurate_negative_verdicts += 1
        elif label == 1 and verdict == 'pos':
          accurate_verdicts += 1
          accurate_positive_verdicts += 1
    print('Accurate %d out of %d total.\n' %
          (accurate_verdicts, total_verdicts))
    print('Accurate poitive %d out of %d total positive.\n' %
          (accurate_positive_verdicts, total_positive_verdicts))
    print('Accurate negative %d out of %d total negative.\n' %
          (accurate_negative_verdicts, total_negative_verdicts))
    print('Estimated total transfer accuracy: %f\n' %
          (float(accurate_verdicts)/float(total_verdicts)))
    print('Estimated positive transfer accuracy: %f\n' %
          (float(accurate_positive_verdicts)/float(total_positive_verdicts)))
    print('Estimated negative transfer accuracy: %f\n' %
          (float(accurate_negative_verdicts)/float(total_negative_verdicts)))


def write_results_to_file(directory, file_name, predict_results):
  with open(os.path.join(directory, file_name), 'w') as f:
    for i in range(0, len(predict_results)):
      if predict_results[i] > PROBABILITY_DISCRIMINATOR_CUTOFF:
        line_string = '%d: pos w/ probability (%f)\n' % (i, predict_results[i])
      else:
        line_string = '%d: neg w/ probability (%f)\n' % (i,
                                                         1.0-predict_results[i])
      f.write(line_string)
  return


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


predict_data = load_dataset_from_feat(os.path.join(
    '..'), 'labeledKerasIMDBBoW.feat', use_for_predictions=True)

predict_data_mhe = weighted_multi_hot_sequences(predict_data)

predicted_results = None
predicted_results = model.predict(predict_data_mhe, verbose=1)


file_name = "model_a.txt"
s = ""
for i in range(0, len(predicted_results)):
    s = s + str(format(float(predicted_results[i][0]), ".8f"))
    if i != len(predicted_results) - 1:
        s = s + "\n"

f = open(file_name, "w")
f.write(s)
f.close()
