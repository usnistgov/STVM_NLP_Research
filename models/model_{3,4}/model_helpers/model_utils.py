import numpy as np
import pandas as pd
import tensorflow_hub as hub
import matplotlib.pyplot as plt

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
    print(classification_report(np.argmax(y_test,axis=1), pred_test))
    return prediction_prob

# Output imdb result to a file
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

def get_history (history):
    # list all data in history
    print(history.history.keys())
    history.history
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    if "val_loss" in history.history:
        plt.plot(history.history['val_loss'])
        plt.title('Training and validation loss')
    else:
        plt.title('Training loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Training loss', 'Validation loss'],)
        plt.show()
        
        
    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    if "val_accuracy" in history.history:
        plt.plot(history.history['val_accuracy'])
        plt.title('Training and validation accuracy')
    else:
        plt.title('Training accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['Training accuracy', 'Validation accuracy'],)
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

        