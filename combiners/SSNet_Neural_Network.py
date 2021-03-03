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
    Jin Honglan
        Affiliation: National Institute of Standards and Technology
        Email: honglan.jin@nist.gov
====================================================================================================================
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.models import Sequential


import numpy as np

EPOCH = 800
BATCH_SIZE = 512
L2_ETA = 0.0039
L2_ETA = [0.0039, 0.041, 0.001]
L1_ETA = 0.0005

def nn(tr_list, imdb_tr_list, te_list, imdb_te_list):
    TR_SAMPLE_SIZE = len(imdb_tr_list)
    TR_PROBA = len(tr_list)

    for idx in range(len(tr_list)):
        assert len(tr_list[idx]) == TR_SAMPLE_SIZE, "train mismatch samples"
    
    train_x = np.zeros([TR_SAMPLE_SIZE, TR_PROBA], dtype=np.float64)
    train_y = np.array([])

    for idx in range(TR_SAMPLE_SIZE):
        ll = imdb_tr_list[idx]
        fn = ll[0]
        label = int(ll[1])

        x = []
        for i in range(TR_PROBA):
            x.append(tr_list[i][fn])

        x_ = np.array(x)
        train_x[idx] = x_
        train_y = np.append(train_y, label)

    model = Sequential()
    model.add(Dense(1, activation=None, use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(L2_ETA[0]),
#                    kernel_regularizer=tf.keras.regularizers.l1(L1_ETA),
                    kernel_constraint=tf.keras.constraints.NonNeg(), input_shape=(TR_PROBA,)))
    model.add(Dense(1, activation="sigmoid"))

    #model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy', 'binary_crossentropy'])

    hist = model.fit(train_x, train_y, epochs=EPOCH, batch_size=BATCH_SIZE,
              shuffle=False, verbose=0)

    weights = model.get_weights()


    #print("trained weights: ", weights)

    # prediction

    TE_SAMPLE_SIZE = len(imdb_te_list)
    TE_PROBA = len(te_list)

    for idx in range(len(te_list)):
        assert len(te_list[idx]) == TE_SAMPLE_SIZE, "test mismatch samples"


    x_predict = np.zeros([len(imdb_te_list), TE_PROBA], dtype=np.float64)
    y_actual = list()

    for idx in range(TE_SAMPLE_SIZE):
        ll = imdb_te_list[idx]
        fn = ll[0]
        label = int(ll[1])

        x = []
        for i in range(TE_PROBA):
            x.append(te_list[i][fn])

        x_ = np.array(x)
        x_predict[idx] = x_
        y_actual.append(label)
    
    y_pred = model.predict(x_predict)

    correct_pred = 0
    wrong_pred = 0

    for idx in range(TE_SAMPLE_SIZE):
        if (y_pred[idx][0] <= 0.5 and y_actual[idx] == 0) or (y_pred[idx][0] > 0.5 and y_actual[idx] == 1):
            correct_pred = correct_pred + 1
        else:
            wrong_pred = wrong_pred + 1
    
    assert (correct_pred + wrong_pred) == TE_SAMPLE_SIZE, "mismatch size"

    _acc = float(correct_pred)/float(TE_SAMPLE_SIZE)
    #print("Accuracy: ", float(correct_pred)/float(TE_SAMPLE_SIZE))
    tr_acc = hist.history['acc'][EPOCH-1] * 100.
    te_acc = _acc * 100.
    return tr_acc, te_acc, weights

