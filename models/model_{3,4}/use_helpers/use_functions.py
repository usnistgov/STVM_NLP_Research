## Universal Sentence Encoder helper functions
import tensorflow_hub as hub
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Dense, Flatten, LSTM, MaxPooling1D, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
## Modelling
def get_model(callable_object, class_num, arch_type):
    vector_size = 512
    hub_layer = hub.KerasLayer(handle=callable_object, output_shape = [vector_size], input_shape = [], dtype = tf.string, trainable = True, name="use_hub_layer")

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
        model.add(tf.keras.layers.Dense(class_num, activation='softmax'))
        
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