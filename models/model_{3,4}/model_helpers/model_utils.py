import numpy as np
import pandas as pd
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import sys

from enum import Enum

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
    report = classification_report(np.argmax(y_test,axis=1), pred_test, digits=4)
    print(report)
    
#   original_stdout = sys.stdout 
#   file name must use length of y_test because x_test has multiple dim
#   evalFile = model_name + '_' + str(len(y_test)) + '_eval.txt'
#   with open(evalFile, 'w') as f:
#       sys.stdout = f # Change the standard output to the file we created.
#       print(results)
#       print(classification_report(np.argmax(y_test,axis=1), pred_test))
#       sys.stdout = original_stdout # Reset the standard output to its original value
        
    evalFile = model_name + '_' + str(len(y_test)) + '_eval'+ '.txt'  
    with open(evalFile, 'w') as f:
        print("The loss and accuracy are:", file=f)
        print(results, file=f)
        print(classification_report(np.argmax(y_test,axis=1), pred_test, digits=4), file=f)
    return prediction_prob

# Output imdb result to a file
def output_result(df_data, file_root_name, run_type, prediction_prob, SPLIT_TRAIN_SIZE):
    df_final = pd.DataFrame(df_data)
    # Get the positive column probabilities
    np_prob = prediction_prob[:,1]
    # Add positive column values to the original dataset
    df_final['prob'] = np_prob
    # Create .csv file including prediction probability, and the file name where the review is from
    result_file_name = file_root_name + ".csv"
    if run_type == 2:
        result_file_name = file_root_name + '_split_' + str(len(df_data)) + '_' + str(SPLIT_TRAIN_SIZE) + '.csv'
#     if RUN_TYPE == 2:
#         if 'test25k' not in file_root_name:
#             file_type = "_train_"
#         else:
#             file_type = "_test_"
#         result_file_name = file_root_name + "_split" + file_type + str(25000 - SPLIT_TRAIN_SIZE) + ".csv"
    if run_type == 3:
        result_file_name = "tmp_" + file_root_name + ".csv"
    df_final.to_csv(result_file_name, index=False, columns = ['prob', 'file'], float_format='%.6f')

def get_history (history):
    # list all data in history
    print(history.history.keys())
    history.history
    
    
    # summarize history for loss
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(history.history['loss'], label='train')
    if "val_loss" in history.history:
        plt.plot(history.history['val_loss'], label='val')
        plt.title('Train and validation loss')
    else:
        plt.title('Train loss')
    plt.legend()
    plt.show()
        
        
    # Summarize history for accuracy
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(history.history['accuracy'], label='train')
    if "val_accuracy" in history.history:
        plt.plot(history.history['val_accuracy'], label='val')
        plt.title('Train and validation accuracy')
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

        