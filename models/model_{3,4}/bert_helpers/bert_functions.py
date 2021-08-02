import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import  Model
from tensorflow.keras.optimizers import Adam
import bert
from tqdm import tqdm
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


pre_trained_bert_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
# Load the pre-trained BERT base model for tokenization
bert_layer_no_trainable = hub.KerasLayer(handle=pre_trained_bert_url, trainable=False, name="bert_layer") 

# Model parameter set
BATCH_SIZE = 8
EPOCHS = 4
#EPOCHS = 2 ## for testing purpose
LRATE = 2e-5
LOSS = 'categorical_crossentropy'
METRICS = 'accuracy'

# Functions for constructing BERT Embeddings: input_ids, input_masks, input_segments and Inputs
MAX_SEQ_LEN = 512 # max sequence length

def get_masks(tokens):
    """Masks: 1 for real tokens and 0 for paddings"""
    return [1]*len(tokens) + [0] * (MAX_SEQ_LEN - len(tokens))
 
def get_segments(tokens):
    """Segments: 0 for the first sequence, 1 for the second"""  
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (MAX_SEQ_LEN - len(tokens))

def get_ids(tokens, tokenizer):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens,)
    input_ids = token_ids + [0] * (MAX_SEQ_LEN - len(token_ids))
    return input_ids

def create_single_input(sentence, tokenizer, max_len):
    """Create an input from a sentence"""
    stokens = tokenizer.tokenize(sentence)
    stokens = stokens[:max_len]
    stokens = ["[CLS]"] + stokens + ["[SEP]"]
 
    ids = get_ids(stokens, tokenizer)
    masks = get_masks(stokens)
    segments = get_segments(stokens)

    return ids, masks, segments
 
def convert_sentences_to_features(sentences, tokenizer):
    """Convert sentences to features: input_ids, input_masks and input_segments"""
    input_ids, input_masks, input_segments = [], [], []
 
    for sentence in tqdm(sentences,position=0, leave=True):
      ids,masks,segments=create_single_input(sentence,tokenizer,MAX_SEQ_LEN-2)
      assert len(ids) == MAX_SEQ_LEN
      assert len(masks) == MAX_SEQ_LEN
      assert len(segments) == MAX_SEQ_LEN
      input_ids.append(ids)
      input_masks.append(masks)
      input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32), 
          np.asarray(input_masks, dtype=np.int32), 
          np.asarray(input_segments, dtype=np.int32)]

# Get Bert tokenizer
def get_bert_tokenizer():
    """Instantiate Tokenizer with vocab"""
    vocab_file=bert_layer_no_trainable.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case=bert_layer_no_trainable.resolved_object.do_lower_case.numpy() 
    tokenizer=bert.bert_tokenization.FullTokenizer(vocab_file,do_lower_case)
    # Get Bert tokenizer
    return tokenizer
   

## Modelling
def get_model():
    # Load the pre-trained BERT base model
    bert_layer = hub.KerasLayer(handle=pre_trained_bert_url, trainable=True, name="bert_layer") 
    # BERT layer three inputs: ids, masks and segments
    input_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_ids")    
    input_masks = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_masks")       
    input_segments = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="segment_ids")
    
    inputs = [input_ids, input_masks, input_segments] # BERT inputs
    pooled_output, sequence_output = bert_layer(inputs) # BERT outputs
    
    # Add a hidden layer
    x = Dense(units=768, activation='relu')(pooled_output)
    x = Dropout(0.1)(x)
 
    # Add output layer
    outputs = Dense(imdb.class_num, activation="softmax", name="predictions")(x)

    # Construct a new model
    model = Model(inputs=inputs, outputs=outputs)
    return model

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

# Give raw input train data, prepare it and get it ready for train and test.
# df_fit_data is the same data as df_data except it is shuffled. This is used to create prob file.
def get_data_ready(df_data):
    data = text, label, df_fit_data = model_utils.get_fit_data(df_data, False, imdb.text_column, imdb.label_column)
    # Prepare model required final input and output 
    x_data = x_input1, x_input2, x_input3 = convert_sentences_to_features(data[0], get_bert_tokenizer())
    y_data = data[1]
    return x_data, y_data, df_fit_data

    
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
        train, test = df_train.iloc[train_idx], df_train.iloc[test_idx]
        i += 1
        if (partial_run and i in run_folds) or not partial_run:
            # check class balance
            print(test.loc[:, [imdb.label_column]].value_counts())

            x_train, y_train, df_fit_train = get_data_ready(train)
            # In SSNet, df_fit_test is used to output prediction probabilities.
            x_test, y_test, df_fit_test = get_data_ready(test)
            # Modelling on top of fine-tuned Bert model
            model = get_model()
            #model.summary()
           
            # Get model name
            root_name = model_utils.get_CV_model_root_name ('bert', i, len(y_train), EPOCHS)
            model_name = root_name + ".h5"
            print(model_name)
            
            history = fit_model(model, x_train, y_train, x_test, y_test)

            # Save the trained model
            model.save(model_name)
            model_utils.get_history(history)

            # Get predictions for test data
            print("Evaluate model: " + model_name)
            prediction_prob, results = model_utils.get_model_performance(model,root_name, x_test, y_test, BATCH_SIZE)

            # Create result file
            result_file_root_name = model_utils.get_CV_prob_root_name (root_name, len(y_test))
            model_utils.output_prob_file(df_fit_test, result_file_root_name, prediction_prob)

            acc = results[1]
            print('> %.3f' % (acc * 100.0))
            # stores scores
            scores.append(acc)
            histories.append(history)
            del([model, train, test])
	
    if (not partial_run):
        model_utils.show_mean_acc_std(scores)
        model_utils.summarize_diagnostics(histories)

# Evaluate all folds and get the summary
def evaluate_k_fold_cv(df_train, n_folds):
    # Get Bert tokenizer
    tokenizer = get_bert_tokenizer()
    
    kfold = KFold(n_folds, shuffle=True, random_state=1)
   
    # loop through each fold and only run evaluation
    i = 0
    scores = list()
    for train_idx, test_idx in kfold.split(df_train):
        i += 1
        train, test = df_train.iloc[train_idx], df_train.iloc[test_idx]
        
        x_test, y_test, df_fit_test = get_data_ready(test)
        
        # Get model name
        root_name = model_utils.get_CV_model_root_name ('bert', i, len(train_idx), EPOCHS)
        model_name = root_name + ".h5"
        print(model_name)

        from tensorflow.keras.models import load_model
        model = load_model(model_name, custom_objects={'KerasLayer':hub.KerasLayer})
        # Set decimal format

        # Get predictions for test data
        prediction_prob, results = model_utils.get_model_performance(model, root_name, x_test, y_test, BATCH_SIZE)
        acc = results[1]

        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)

    # print summary
    model_utils.show_mean_acc_std(scores)


