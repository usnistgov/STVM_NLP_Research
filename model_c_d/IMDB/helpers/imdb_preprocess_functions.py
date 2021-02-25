import pandas as pd
from tensorflow.keras.utils import to_categorical
import tensorflow_hub as hub
import numpy as np


# Columns to use
text_column = 'review'
label_column = 'label'
file_column = 'file'
probability_column = 'prob'

# Read IMDB .csv file, return train and test datasets in Dataframe format after some preprocessing
def get_imdb_df_data(csv_file):
    #Read the IMDB dataset into Pandas dataframe. This includes training, test, and unsupervised data. Remove unsupervised data.
    df_master = pd.read_csv(csv_file)
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


# Given a dataframe data, create model required fitted data
def get_fit_data(df, shuffle):
    df_shuffled = df
    X_fit = df[text_column]
    y_fit = to_categorical(df[label_column].values)
    if shuffle == True:
        # Shuffle the data
        df_shuffled = df.sample(frac=1, random_state=0)
        X_fit = df_shuffled[text_column]
        y_fit = to_categorical(df_shuffled[label_column].values)
    return X_fit, y_fit, df_shuffled

# Get a model's performance
def get_model_performance(model_name, x_test, y_test, BATCH_SIZE):
    # Load the pretrained nlp_model
    from tensorflow.keras.models import load_model
    new_model = load_model(model_name, custom_objects={'KerasLayer':hub.KerasLayer})
    # Set decimal format
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    prediction_prob = new_model.predict(x_test)
    results = new_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print(results)
    # Predict on test dataset
    from sklearn.metrics import classification_report
    #pred_test = np.argmax(new_model.predict(X_test), axis= 1)
    pred_test = np.argmax(prediction_prob, axis= 1)
    print(classification_report(np.argmax(y_test,axis=1), pred_test))
    return prediction_prob

# Output result to a file
def output_result(df_data, file_root_name, RUN_TYPE, prediction_prob, SPLIT_TRAIN_SIZE):
    df_final = pd.DataFrame(df_data)
    # Get the positive column probabilities
    np_prob = prediction_prob[:,1]
    # Add positive column values to the original dataset
    df_final['prob'] = np_prob
    # Create .csv file including prediction probability, and the file name where the review is from
    result_file_name = file_root_name + "_25k.csv"
    if RUN_TYPE == 2:
        if SPLIT_TRAIN_SIZE > 0:
            file_type = "_train_"
        else:
            file_type = "_test_"
        result_file_name = file_root_name + "_split_" + file_type + str(25000 - SPLIT_TRAIN_SIZE) + ".csv"
    if RUN_TYPE == 3:
        result_file_name = "tmp_" + file_root_name + ".csv"
    df_final.to_csv(result_file_name, index=False, columns = ['prob', 'file'], float_format='%.6f')