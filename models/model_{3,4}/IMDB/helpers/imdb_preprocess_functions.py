# This module provides functions to imdb data

import pandas as pd
from tensorflow.keras.utils import to_categorical
import tensorflow_hub as hub
import numpy as np
import sys

sys.path.append("../model_helpers")
import model_utils as model_utils
import importlib
importlib.reload(model_utils)

# IMDB data location
imdb_data = '/wrk/hnj9/imdb_data/' + 'imdb_master.csv'
class_num = 2


# Columns to use
text_column = 'review'
label_column = 'label'
file_column = 'file'
probability_column = 'prob'

# Read IMDB .csv file, return train and test datasets in Dataframe format after some preprocessing
def get_imdb_df_data():
    #Read the IMDB dataset into Pandas dataframe. This includes training, test, and unsupervised data. Remove unsupervised data.
    df_master = pd.read_csv(imdb_data)
    df_master = df_master[df_master[label_column] != 'unsup']
    df_master[text_column] = df_master[text_column].str.replace("<br />", " ")
    # Convert labels to class digits
    df_master[label_column].replace('pos',1.,inplace=True)
    df_master[label_column].replace('neg',0.,inplace=True)
    # Read the IMDB training dataset into Pandas dataframe
    df_train = df_master[df_master['type'] == 'train']
    print("The number of rows and columns in the training dataset is: {}".format(df_train.shape))
    # Identify missing values in train dataset
    train_missing = df_train.apply(lambda x: sum(x.isnull()), axis=0)
    print ('Missing values in train dataset:')
    print(train_missing)
    # Check the target class balance
    train_class_balance = df_train[label_column].value_counts()
    print ('Check train class balance')
    print (train_class_balance)
    df_test = df_master[df_master['type'] == 'test']
    print("The number of rows and columns in the test dataset is: {}".format(df_test.shape))
    # Identify missing values in test dataset
    test_missing = df_test.apply(lambda x: sum(x.isnull()), axis=0)
    print ('Missing values in test dataset:')
    print(test_missing)
    # Check the test dataset class balance
    test_class_balance = df_test[label_column].value_counts()
    print ('Check test class balance')
    print (test_class_balance)
    return df_train, df_test

# Generate IMDB training model name depending on run type
def get_model_name (run_type, output_dir, model_type, split_train_size, epoch_num):
    model_file_prefix = ''
    root_name = output_dir + 'model_' + model_type
    if run_type == 1:
        model_file_prefix = root_name + '_25000_'
    if run_type == 2:
        model_file_prefix = root_name + '_' + str(split_train_size) + '_'
    if run_type == 3:
        model_name = root_name +'_'+ 'epoch_' + str(epoch_num) + "_tmp.h5"
    else:
        model_name = model_file_prefix + 'epoch_' + str(epoch_num) + ".h5"
    return model_name, root_name

# Give a whole dataset, extract specified size of data for testing run. Make sure the data is balanced in classes
def get_test_run_data(df_train_data, df_test_data, run_size):
    # Shuffle the dataset to balance classes before getting TESTING run data, which is small enough for quick run
    df_train = df_train_data.sample(frac=1, random_state=0)[:run_size]
    df_test = df_test_data.sample(frac=1, random_state=0)[:run_size]
#     df_test = df_test[:run_size]
    print("The number of rows and columns in the training dataset is: {}".format(df_train.shape))
    print("The number of rows and columns in the test dataset is: {}".format(df_test.shape))
    # Check the test dataset class balance
    train_class_balance = df_train[label_column].value_counts()
    print ('Check rain class balance')
    print (train_class_balance)
    # Check the test dataset class balance
    test_class_balance = df_test[label_column].value_counts()
    print ('Check test class balance')
    print (test_class_balance)
    return df_train, df_test