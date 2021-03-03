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

'''
This is master file that runs all the three combiners proposed in the paper. 
Use following snippet to run all the three combiners: python SSNet_predictions.py
Please note that this code has tensoflow dependencies.
'''

import tensorflow as tf
import math
import re

import pandas as pd
import numpy as np
import random
import os

from SSNet_Neural_Network import nn
from SSNet_Bayesian_Decision import bayesian_decision
from SSNet_Heuristic_Hybrid import heuristic_hybrid


imdb_5ktr = 'imdb_train_5k.csv'
model_a_tr = 'model_a_5ktrain.csv'
model_b_tr = 'model_b_5ktrain.csv'
model_c_tr = 'model_c_bert_result_train_5k.csv'
model_d_tr = 'model_d_use_result_train_5k.csv'

model_a_te = 'model_a_25ktest.csv'
model_b_te = 'model_b_25ktest.csv'
model_c_te = 'model_c_bert_result_test_25k.csv'
model_d_te = 'model_d_use_result_test_25k.csv'


def get_training_dict_threshold(split):
    training_dict = dict()

    if split == "5K":
        training_dict["5K"] = [
            [model_a_tr, model_b_tr, model_c_tr, model_d_tr], [
                model_a_te, model_b_te, model_c_te, model_d_te]
        ]
        
    return training_dict

def get_training_dict(split):
    training_dict = dict()
    if split == "5K":
        training_dict["model_{1,2}"] = [
            [model_a_tr, model_b_tr], [model_a_te, model_b_te]
        ]
        training_dict["model_{1,3}"] = [
            [model_a_tr, model_c_tr], [model_a_te, model_c_te]
        ]
        training_dict["model_{1,4}"] = [
            [model_a_tr, model_d_tr], [model_a_te, model_d_te]
        ]
        training_dict["model_{2,3}"] = [
            [model_b_tr, model_c_tr], [model_b_te, model_c_te]
        ]
        training_dict["model_{2,4}"] = [
            [model_b_tr, model_d_tr], [model_b_te, model_d_te]
        ]
        training_dict["model_{3,4}"] = [
            [model_c_tr, model_d_tr], [model_c_te, model_d_te]
        ]

        training_dict["model_{1,2,3}"] = [
            [model_a_tr, model_b_tr, model_c_tr], [model_a_te, model_b_te, model_c_te]
        ]
        training_dict["model_{1,2,4}"] = [
            [model_a_tr, model_b_tr, model_d_tr], [model_a_te, model_b_te, model_d_te]
        ]
        training_dict["model_{1,3,4}"] = [
            [model_a_tr, model_c_tr, model_d_tr], [model_a_te, model_c_te, model_d_te]
        ]
        training_dict["model_{2,3,4}"] = [
            [model_b_tr, model_c_tr, model_d_tr], [model_b_te, model_c_te, model_d_te]
        ]

        training_dict["model_{1,2,3,4}"] = [
            [model_a_tr, model_b_tr, model_c_tr, model_d_tr], [
                model_a_te, model_b_te, model_c_te, model_d_te]
        ]

    return training_dict



imdb_25k_list = list()
for file_name in os.listdir('../test/pos'):
    if file_name != '.DS_Store':
        imdb_25k_list.append([file_name, str(1)])

for file_name in os.listdir('../test/neg'):
    if file_name != '.DS_Store':
        imdb_25k_list.append([file_name, str(0)])

SAMPLE_SPLIT = ["5K"]

for split in SAMPLE_SPLIT:
    print("Sample Split: ", split)
    imdb_list = list()
    training_dict = None
    training_dict_threshold = None

    if split == "5K":
        df_imdb_tr = pd.read_csv(imdb_5ktr)
        for index in df_imdb_tr.index:
            file_name = str(df_imdb_tr['file'][index])
            label = int(df_imdb_tr['label'][index])
            imdb_list.append([file_name, str(label)])

        random.shuffle(imdb_list)
        training_dict = get_training_dict(split)
        training_dict_threshold = get_training_dict_threshold(split)

        random.shuffle(imdb_list)
        training_dict = get_training_dict(split)
        training_dict_threshold = get_training_dict_threshold(split)


    acc_dict_nn = dict()
    acc_dict_bdc = dict()
    
    for k, v in training_dict.items():

        tr_list = list()
        te_list = list()

        for i in range(len(v[0])):
            df = pd.read_csv(v[0][i])
            df_dict = dict()

            for idx in df.index:
                file_name = str(df['file'][idx])
                proba = float(df['prob'][idx])
                df_dict[file_name] = proba
            
            tr_list.append(df_dict)

        for i in range(len(v[1])):
            df = pd.read_csv(v[1][i])
            df_dict = dict()

            for idx in df.index:
                file_name = str(df['file'][idx])
                proba = float(df['prob'][idx])
                df_dict[file_name] = proba
            
            te_list.append(df_dict)

        
        assert len(tr_list) == len(te_list), "train and test samples mismatch ...."
        tr_acc = -1.
        te_acc = -1.
        while True:
            tr_acc, te_acc, weights = nn(tr_list=tr_list, imdb_tr_list=imdb_list,
                te_list=te_list, imdb_te_list=imdb_25k_list)
            
            if weights[0][0] == 0. or weights[0][1] == 0.:
                print("bad event ...., training again")
                print("\t" +k)
            else:
                break

        acc_dict_nn[k] = [tr_acc, te_acc]
        
        acc_dict_bdc[k] = bayesian_decision(tr_list=tr_list, imdb_tr_list=imdb_list,
                                         te_list=te_list, imdb_te_list=imdb_25k_list)

    
    for k, v in training_dict_threshold.items():
        tr_list = list()
        te_list = list()

        for i in range(len(v[0])):
            df = pd.read_csv(v[0][i])
            df_dict = dict()

            for idx in df.index:
                file_name = str(df['file'][idx])
                proba = float(df['prob'][idx])
                df_dict[file_name] = proba

            tr_list.append(df_dict)

        for i in range(len(v[1])):
            df = pd.read_csv(v[1][i])
            df_dict = dict()

            for idx in df.index:
                file_name = str(df['file'][idx])
                proba = float(df['prob'][idx])
                df_dict[file_name] = proba

            te_list.append(df_dict)

    hh_dict = heuristic_hybrid(tr_list=tr_list, imdb_tr_list=imdb_list,
                    te_list=te_list, imdb_te_list=imdb_25k_list)

    
    #print("Training Complete: ")
    print("Neural Network Combiner: ")
    for k, v in acc_dict_nn.items():
        print("\t" +k +": training accuracy = " +str(v[0]) + ", test accuracy = " +str(v[1]))
    
    print("\n")
    print("Bayesian Decision Rule Combiner: ")
    for k, v in acc_dict_bdc.items():
        print("\t" +k)
        for i, j in v.items():
            print("\t\t" +i +": training accuracy = " +str(j[0]) +", test accuracy = " +str(j[1]))
    
    print("\n")
    print("Heuristic-Hybrid Combiner: ")
    for k, v in hh_dict.items():
        print("Base:", k)
        for index in range(len(v)):
            print("\t\t", v[index])
        print("\n")