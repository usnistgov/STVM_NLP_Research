# This module provides functions to model independent generic features

import numpy as np
import pandas as pd
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import KFold

from numpy import mean
from numpy import std

from enum import Enum


# Set output folder in non-quote file system area
root_output_dir = "../model_ouput/"
# Get a model's performance
def get_model_performance(model, root_name, x_test, y_test, BATCH_SIZE):
    # Set decimal format
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    prediction_prob = model.predict(x_test)
 
    results = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print("results are: ")
    print(results)
  
    # Predict on test dataset
    from sklearn.metrics import classification_report
    
    #pred_test = np.argmax(new_model.predict(X_test), axis= 1)
    pred_test = np.argmax(prediction_prob, axis=1)
    report = classification_report(np.argmax(y_test,axis=1), pred_test, digits=4)
    print(report)
    
    evalFile = get_CV_eval_root_name (root_name, len(y_test)) + '.txt'
    with open(evalFile, 'w') as f:
        print("The loss and accuracy are:", file=f)
        print(results, file=f)
        print(classification_report(np.argmax(y_test,axis=1), pred_test, digits=4), file=f)
    
    return prediction_prob, results

    
# Output imdb result to a file
def output_prob_file(df_data, file_root_name, prediction_prob):
    df_final = pd.DataFrame(df_data)
    # Get the positive column probabilities
    # print(prediction_prob)
    np_prob = prediction_prob[:,1]
    # Add positive column values to the original dataset
    df_final['prob'] = np_prob
    # Create .csv file including prediction probability, and the file name where the review is from
    result_file_name = file_root_name + ".csv"
    df_final.to_csv(result_file_name, index=False, columns = ['prob', 'file'], float_format='%.6f')


def get_history (history):
    # list all data in history
    print(history.history.keys())
    history.history
    
    
    # summarize history for loss
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(history.history['loss'], color='blue', label='train')
    if "val_loss" in history.history:
        plt.plot(history.history['val_loss'], color='orange', label='val')
        plt.title('Train vs Validation Loss')
    else:
        plt.title('Train loss')
    plt.legend()
    plt.show()
        
        
    # Summarize history for accuracy
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    if "val_accuracy" in history.history:
        plt.plot(history.history['val_accuracy'], color='orange', label='val')
        plt.title('Train vs Validation Accuracy')
    else:
        plt.title('Train accuracy')
    plt.legend()
    plt.show()
        

# Define run types
class RunType(Enum):
    FULL = 1
    SPLIT = 2
    SHORT = 3
    
    def describe(self):
        description = "Full run"
        if (self is self.SPLIT):
            description = 'Train data is split to two parts, first part is used in training, and second part is used in testing'
        if (self is self.SHORT):
            description = 'Short run using small number of samples to train and test to see if the work flow works correctly'
        return self.name, self.value, description
    
    @classmethod
    def get_run_detail(cls):
        return cls.FULL.describe(), cls.SPLIT.describe(), cls.SHORT.describe()

    
# Given a dataframe data, create model required fitted data
def get_fit_data(df, shuffle, text_column, label_column):
    df_final = df
    if shuffle == True:
        df_final = df.sample(frac=1, random_state=0)
    X_fit = df_final[text_column]
    #y_fit, class_num = get_categorical(df_final.iloc[:,-1:].values.ravel()) # can't assume last column as the categorial variable
    y_fit, class_num = get_categorical(df_final[label_column])
    return X_fit, y_fit, df_final


# Create one hot encoding for categorical variable, and number of classes to clasify
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
def get_categorical(Y):
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_data = to_categorical(encoded_Y)
    return y_data, y_data.shape[1]

# evaluate a model using k-fold cross-validation
def evaluate_model(df, n_folds, model):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(df):
		train, test = df_train.iloc[train_idx], df_train.iloc[test_idx]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories

# Get model name for k-fold cross-validation without extention. CV for cross-validation.
# Modal name composition rule: 1. model_output (/wrk/hnj9/model_output/), 2. model_type (bert/use), 3. ith fold number, 
# 4. train data size and 5. epoch number
# Get model root name (extention: .h5)
def get_CV_model_root_name (model_type, i, train_size, epoch_num):
    model_dir = root_output_dir + model_type + '/'
    model_file_postfix = "_" + "fold_" + str(i) + '_tr' + str(train_size) + '_' + str(epoch_num)
    root_name = model_dir + 'model_' + model_type + model_file_postfix
    return root_name

# Get evaluation file root name  (_eval.txt)
def get_CV_eval_root_name (model_root_name, test_size):
    eval_root_name = model_root_name + '_te' + str(test_size) + '_eval'
    return eval_root_name

# Get probability output file root (name _prob.csv)
def get_CV_prob_root_name (model_root_name, test_size):
    prob_root_name = model_root_name + '_te' + str(test_size) + '_prob'
    return prob_root_name


# For cross-validation. loop through each fold, run evaluation with the test data provided and create run results and evaluation results
# Assume the models in all folds are already built
def evaluate_test_dataset_k_fold(x_test, y_test, n_folds, model_type, EPOCHS, BATCH_SIZE, df_fit_test, train_size):
    scores = list()
    for i in range(1, n_folds+1):
        # Get model name
        root_name = get_CV_model_root_name (model_type, i, train_size, EPOCHS)
        model_name = root_name + ".h5"
        print(model_name)
        
        model = load_model(model_name, custom_objects={'KerasLayer':hub.KerasLayer})
        
        prediction_prob, results = get_model_performance(model, root_name, x_test, y_test, BATCH_SIZE)
        acc = results[1]
        
        result_root_name = get_CV_prob_root_name (root_name, len(y_test))
        output_prob_file(df_fit_test, result_root_name, prediction_prob)
        
        print('> %.3f' % (float(acc) * 100.0))
        scores.append(acc)

    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()

    
# plot diagnostic learning curves
def summarize_diagnostics(histories):
    # plot loss
    plt.subplot(2, 1, 1)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Train vs Validation Loss')
    
    for i in range(len(histories)):
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='val')
    plt.show()
    
    # plot accuracy
    plt.subplot(2, 1, 2)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title('Train vs Validation Classification Accuracy')
    for i in range(len(histories)):
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='val')
    plt.show()

# Display mean accuracy and std
def show_mean_acc_std(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()
    