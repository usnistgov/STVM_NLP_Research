## Universal Sentence Encoder helper functions
import tensorflow_hub as hub
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Dense, Flatten, LSTM, MaxPooling1D, Bidirectional
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import  Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import sys

from sklearn.model_selection import KFold
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt

sys.path.append("../model_helpers")
import model_utils as model_utils
sys.path.append("./helpers")
import imdb_preprocess_functions as imdb

import importlib
importlib.reload(model_utils)
importlib.reload(imdb)


use_model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
#use_model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"

# Model parameter set
BATCH_SIZE = 5
EPOCHS = 7
#EPOCHS = 2 # testing purpose
# LRATE = 2e-4
LRATE = 2e-5 # for when batch size is very small like 5
LOSS = 'categorical_crossentropy'
METRICS = 'accuracy'

vector_size = 512

## Modelling
def get_model(arch_type):
    hub_layer = hub.KerasLayer(handle=use_model_url, output_shape = [vector_size], input_shape = [], dtype = tf.string, trainable = True, name="use_hub_layer")
    model = tf.keras.Sequential()
    
    if arch_type == 1:
        model.add(hub_layer)
        model.add(tf.keras.layers.Input(shape=(512,)))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(class_num, activation="softmax", name="predictions"))
    
    elif arch_type == 2:
        model.add(hub_layer)
        model.add(
        tf.keras.layers.Dense(
            units=256,
            #input_shape=(512, ),
            activation='relu'
          )
        )
        model.add(
        tf.keras.layers.Dropout(rate=0.5)
        )
        model.add(
        tf.keras.layers.Dense(
            units=128,
            activation='relu'
            )
        )
        model.add(
          tf.keras.layers.Dropout(rate=0.5)
        )
        model.add(tf.keras.layers.Dense(imdb.class_num, activation='softmax'))
        
    elif arch_type == 3:
        model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same',
                 input_shape=(vector_size, 1)))
        model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
        model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=3))

        model.add(Bidirectional(LSTM(512, dropout=0.2, recurrent_dropout=0.3)))

        model.add(Dense(512, activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dropout(0.25))
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dropout(0.25))

        model.add(Dense(class_num, activation='softmax'))

    return model

# Give raw input train data, prepare it and get it ready for train and test.
# df_fit_data is the same data as df_data except it is shuffled. This is used to create prob file.
def get_data_ready(df_data):
    data = text, label, df_fit_data = model_utils.get_fit_data(df_data, False, imdb.text_column, imdb.label_column)
    # Prepare model required final input and output 
    x_data = data[0]
    y_data = data[1]
    return x_data, y_data, df_fit_data

# Fit model with the train data
def fit_model(model, x_train, y_train, x_test, y_test):
    #Use Adam optimizer to minimize the categorical_crossentropy loss
    opt = Adam(learning_rate=LRATE)
    model.compile(optimizer=opt, 
          loss=LOSS,
          metrics=[METRICS])

    # Fit the data to the model
    history = model.fit(x_train, y_train,
                validation_data=(x_test, y_test),
                #validation_split=0.2, # Important: in production, use all training data
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                #callbacks=[tensorboard_callback],
                verbose = 1)    
    return history



# Full run of k-fold cross-validation, or if *idx is present, run folds specified in *idx
def run_k_fold_cv(df_train, n_folds, *idx):
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    partial_run = bool(idx)
    if partial_run:
        # check if a specific fold(s) are specified to run
        run_folds = []
        for x in idx:
            run_folds.append(x)
        run_text = "Fold {0} will be run.".format(run_folds)
        print(run_text)
    
    # loop through each fold, build models for each fold, evaluate performance of each fold and output run and evaluation results
    scores, histories = list(), list()
    i = 0
    for train_idx, test_idx in kfold.split(df_train):
        i += 1
        train, test = df_train.iloc[train_idx], df_train.iloc[test_idx]
        if (partial_run and i in run_folds) or not partial_run:

            # check class balance
            print(test.loc[:, [imdb.label_column]].value_counts())

            x_train, y_train, df_fit_train = get_data_ready(train)
            
            x_test, y_test, df_fit_test = get_data_ready(test)
            
           
            # Get model name
            root_name = model_utils.get_CV_model_root_name ('use', i, len(y_train), EPOCHS)
            model_name = root_name + ".h5"
            print("Current model: " + model_name)
            
            model = get_model(2)
            # model.summary()
            
            history = fit_model(model, x_train, y_train, x_test, y_test)

            # Save the trained model
            model.save(model_name)
            model_utils.get_history(history)

        #     plt.pcolormesh(y_test)
        #     plt.show()

            # Get predictions for test data
            print("Evaluate model: " + model_name)
            prediction_prob, results = model_utils.get_model_performance(model, root_name, x_test, y_test, BATCH_SIZE)

            # Create result file
            result_file_root_name = model_utils.get_CV_prob_root_name (root_name, len(y_test))
            model_utils.output_prob_file(df_fit_test, result_file_root_name, prediction_prob)

            acc = results[1]
            print('> %.3f' % (acc * 100.0))
            # stores scores
            scores.append(acc)
            histories.append(history)
	
    if (not partial_run):
        model_utils.summarize_diagnostics(histories)
        model_utils.show_mean_acc_std(scores)

# Evaluate all folds and get the summary
def evaluate_k_fold_cv(df_train, n_folds):
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # loop through each fold and only run evaluation
    i = 0
    scores = list()
    for train_idx, test_idx in kfold.split(df_train):
        i += 1
        train, test = df_train.iloc[train_idx], df_train.iloc[test_idx]
        
        x_test, y_test, df_fit_test = get_data_ready(test)
        
        # Get model name
        root_name = model_utils.get_CV_model_root_name ('use', i, len(train_idx), EPOCHS)
        model_name = root_name + ".h5"
        print(model_name)

        from tensorflow.keras.models import load_model
        model = load_model(model_name, custom_objects={'KerasLayer':hub.KerasLayer})

        # Get predictions for test data
        prediction_prob, results = model_utils.get_model_performance(model, root_name, x_test, y_test, BATCH_SIZE)
        acc = results[1]

        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)

    # print summary
    model_utils.show_mean_acc_std(scores)