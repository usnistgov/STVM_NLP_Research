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

from itertools import combinations

import math
import re

import pandas as pd
import numpy as np
import random
import os
import sys

from SSNet_Neural_Network import dense
from SSNet_Bayesian_Decision import bayesian_decision
from SSNet_Heuristic_Hybrid import heuristic_hybrid

iterations = 30 # each iteration takes approx 5-10 mins

models = ["model_1", "model_2", "model_3", "model_4"]
SAMPLE_SPLIT = ["5KFold1", "5KFold2", "5KFold3", "5KFold4", "5KFold5"]

base_dir = '../data/'

#train folds
fold1_5ktr = base_dir + 'tr_te_5k_fold_1.csv'
fold2_5ktr = base_dir + 'tr_te_5k_fold_2.csv'
fold3_5ktr = base_dir + 'tr_te_5k_fold_3.csv'
fold4_5ktr = base_dir + 'tr_te_5k_fold_4.csv'
fold5_5ktr = base_dir + 'tr_te_5k_fold_5.csv'

# fold 1 csv files
model_1_tr_Fold1 = base_dir + "model_1_fold1_5ktrain.csv"
model_2_tr_Fold1 = base_dir + "model_2_fold1_5ktrain.csv"
model_3_tr_Fold1 = base_dir + "model_3_fold1_5ktrain.csv"
model_4_tr_Fold1 = base_dir + "model_4_fold1_5ktrain.csv"

model_1_te_Fold1 = base_dir + "model_1_fold1_25ktest.csv"
model_2_te_Fold1 = base_dir + "model_2_fold1_25ktest.csv"
model_3_te_Fold1 = base_dir + "model_3_fold1_25ktest.csv"
model_4_te_Fold1 = base_dir + "model_4_fold1_25ktest.csv"

# fold 2 csv files
model_1_tr_Fold2 = base_dir + "model_1_fold2_5ktrain.csv"
model_2_tr_Fold2 = base_dir + "model_2_fold2_5ktrain.csv"
model_3_tr_Fold2 = base_dir + "model_3_fold2_5ktrain.csv"
model_4_tr_Fold2 = base_dir + "model_4_fold2_5ktrain.csv"

model_1_te_Fold2 = base_dir + "model_1_fold2_25ktest.csv"
model_2_te_Fold2 = base_dir + "model_2_fold2_25ktest.csv"
model_3_te_Fold2 = base_dir + "model_3_fold2_25ktest.csv"
model_4_te_Fold2 = base_dir + "model_4_fold2_25ktest.csv"


# fold 3 csv files
model_1_tr_Fold3 = base_dir + "model_1_fold3_5ktrain.csv"
model_2_tr_Fold3 = base_dir + "model_2_fold3_5ktrain.csv"
model_3_tr_Fold3 = base_dir + "model_3_fold3_5ktrain.csv"
model_4_tr_Fold3 = base_dir + "model_4_fold3_5ktrain.csv"

model_1_te_Fold3 = base_dir + "model_1_fold3_25ktest.csv"
model_2_te_Fold3 = base_dir + "model_2_fold3_25ktest.csv"
model_3_te_Fold3 = base_dir + "model_3_fold3_25ktest.csv"
model_4_te_Fold3 = base_dir + "model_4_fold3_25ktest.csv"


# fold 4 csv files
model_1_tr_Fold4 = base_dir + "model_1_fold4_5ktrain.csv"
model_2_tr_Fold4 = base_dir + "model_2_fold4_5ktrain.csv"
model_3_tr_Fold4 = base_dir + "model_3_fold4_5ktrain.csv"
model_4_tr_Fold4 = base_dir + "model_4_fold4_5ktrain.csv"

model_1_te_Fold4 = base_dir + "model_1_fold4_25ktest.csv"
model_2_te_Fold4 = base_dir + "model_2_fold4_25ktest.csv"
model_3_te_Fold4 = base_dir + "model_3_fold4_25ktest.csv"
model_4_te_Fold4 = base_dir + "model_4_fold4_25ktest.csv"


# fold 5 csv files
model_1_tr_Fold5 = base_dir + "model_1_fold5_5ktrain.csv"
model_2_tr_Fold5 = base_dir + "model_2_fold5_5ktrain.csv"
model_3_tr_Fold5 = base_dir + "model_3_fold5_5ktrain.csv"
model_4_tr_Fold5 = base_dir + "model_4_fold5_5ktrain.csv"

model_1_te_Fold5 = base_dir + "model_1_fold5_25ktest.csv"
model_2_te_Fold5 = base_dir + "model_2_fold5_25ktest.csv"
model_3_te_Fold5 = base_dir + "model_3_fold5_25ktest.csv"
model_4_te_Fold5 = base_dir + "model_4_fold5_25ktest.csv"


def gen_model_combinations(models):
    model_combinations = list()
    for index in range(len(models)):
        if index < 1:
            pass
        else:
            model_combinations = model_combinations + list(
                combinations(models, index + 1))

    return model_combinations


def get_prediction_dict(f):
    model_combinations = gen_model_combinations(models)

    training_dict = dict()
    for combination in model_combinations:
        key = "model_{"
        list_tr_sample = list()
        list_te_sample = list()
        for item in combination:
            key = key + item.split("_")[1] + ","
            list_tr_sample.append(globals()[item + "_tr_Fold" + f[-1]])
            list_te_sample.append(globals()[item + "_te_Fold" + f[-1]])

        key = key[:-1] + "}"

        training_dict[key] = [list_tr_sample, list_te_sample]
    return training_dict


def get_prediction_th_dict(f):
    training_dict = dict()
    list_tr_sample = list()
    list_te_sample = list()
    for model in models:
        list_tr_sample.append(globals()[model + "_tr_Fold" + f[-1]])
        list_te_sample.append(globals()[model + "_te_Fold" + f[-1]])
    training_dict[f] = [list_tr_sample, list_te_sample]
    return training_dict


imdb_25k_list = list()
for file_name in os.listdir("../test/pos"):
    if file_name != '.DS_Store':
        imdb_25k_list.append([file_name, str(1)])

for file_name in os.listdir("../test/neg"):
    if file_name != '.DS_Store':
        imdb_25k_list.append([file_name, str(0)])


acc_dense_dict = dict()
acc_bayesian_dict = dict()
acc_threshold_dict = dict()

##initilize dicts keys
dict_keys_list = gen_model_combinations(models)
for index in range(len(dict_keys_list)):
    key = "model_{"
    for item in dict_keys_list[index]:
        key = key + item.split("_")[1] + ","
    key = key[:-1] + "}"
    dict_keys_list[index] = key

print(dict_keys_list)
for index in range(len(dict_keys_list)):
    acc_dense_dict[dict_keys_list[index]] = [
        list(), list()
    ]  # train test accuracy list
    acc_bayesian_dict[dict_keys_list[index]] = [
        {'max': [list(), list()]}, {'avg': [list(), list()]}, {
            'sum': [list(), list()]}, {'maj': [list(), list()]}
    ]

base_model = ['model_{1}', 'model_{2}', 'model_{3}', 'model_{4}']

for index in range(len(base_model)):
    ll = list()
    for i in range(16):
        dd = dict()
        dd['tr_acc'] = list()
        dd['te_acc'] = list()
        dd['th'] = list()
        ll.append(dd)

    acc_threshold_dict[base_model[index]] = ll


for split in SAMPLE_SPLIT:
    print("=> Fold Number: ", split)
    imdb_list = list()
    training_dict = None
    training_dict_threshold = None

    if split == "5KFold1":
        df_imdb_tr = pd.read_csv(fold1_5ktr)
    if split == "5KFold2":
        df_imdb_tr = pd.read_csv(fold2_5ktr)
    if split == "5KFold3":
        df_imdb_tr = pd.read_csv(fold3_5ktr)
    if split == "5KFold4":
        df_imdb_tr = pd.read_csv(fold4_5ktr)
    if split == "5KFold5":
        df_imdb_tr = pd.read_csv(fold5_5ktr)

    for index in df_imdb_tr.index:
        file_name = str(df_imdb_tr['file'][index])
        label = int(df_imdb_tr['label'][index])
        imdb_list.append([file_name, str(label)])

    random.shuffle(imdb_list)
    training_dict = get_prediction_dict(split)
    training_dict_threshold = get_prediction_th_dict(split)

    acc_dict_mj = dict()

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

        assert len(tr_list) == len(
            te_list), "train and test samples mismatch ...."

        tr_acc_list = []
        te_acc_list = []
        for runs in range(iterations):
            print("\tdense iteration number: " +
                  str(runs) + " for model combination: " + k)
            tr_acc = -1.
            te_acc = -1.
            while True:
                tr_acc, te_acc, weights = dense(tr_list=tr_list, imdb_tr_list=imdb_list,
                                                te_list=te_list, imdb_te_list=imdb_25k_list)

                if weights[0][0] == 0. or weights[0][1] == 0.:
                    print("bad event ...., training again")
                    print("\t" + k)
                else:
                    break
            tr_acc_list.append(tr_acc)
            te_acc_list.append(te_acc)

        acc_dense_dict[k][0] = acc_dense_dict[k][0] + tr_acc_list
        acc_dense_dict[k][1] = acc_dense_dict[k][1] + te_acc_list

        ll_max_tr = list()
        ll_max_te = list()
        ll_avg_tr = list()
        ll_avg_te = list()
        ll_sum_tr = list()
        ll_sum_te = list()
        ll_maj_tr = list()
        ll_maj_te = list()

        print("\tbayesian decision iteration number: " +
              str(runs) + " for model combination: " + k)

        acc_dict_bdc = bayesian_decision(tr_list=tr_list, imdb_tr_list=imdb_list,
                                         te_list=te_list, imdb_te_list=imdb_25k_list)

        for k1, v1 in acc_dict_bdc.items():
            if k1 == 'max':
                ll_max_tr.append(v1[0])
                ll_max_te.append(v1[1])
            elif k1 == 'avg':
                ll_avg_tr.append(v1[0])
                ll_avg_te.append(v1[1])
            elif k1 == 'sum':
                ll_sum_tr.append(v1[0])
                ll_sum_te.append(v1[1])
            elif k1 == 'maj':
                ll_maj_tr.append(v1[0])
                ll_maj_te.append(v1[1])
            else:
                print("error....")
                sys.exit()

        acc_kk = acc_bayesian_dict[k]
        acc_kk[0]['max'][0] = acc_kk[0]['max'][0] + ll_max_tr
        acc_kk[0]['max'][1] = acc_kk[0]['max'][1] + ll_max_te

        acc_kk[1]['avg'][0] = acc_kk[1]['avg'][0] + ll_avg_tr
        acc_kk[1]['avg'][1] = acc_kk[1]['avg'][1] + ll_avg_te

        acc_kk[2]['sum'][0] = acc_kk[2]['sum'][0] + ll_sum_tr
        acc_kk[2]['sum'][1] = acc_kk[2]['sum'][1] + ll_sum_te

        if len(ll_maj_tr) > 0:
            acc_kk[3]['maj'][0] = acc_kk[3]['maj'][0] + ll_maj_tr
            acc_kk[3]['maj'][1] = acc_kk[3]['maj'][1] + ll_maj_te

        acc_bayesian_dict[k] = acc_kk

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

    print("\theuristic hybrid iteration number: " +
          str(runs))
    th_dict = heuristic_hybrid(tr_list=tr_list, imdb_tr_list=imdb_list,
                               te_list=te_list, imdb_te_list=imdb_25k_list)

    for k1, v1 in th_dict.items():
        for index in range(len(v1)):
            dd = v1[index]
            _tr_acc = dd['tr_acc']
            _te_acc = dd['te_acc']
            _th = dd['th']
            acc_threshold_dict[k1][index]['tr_acc'].append(_tr_acc)
            acc_threshold_dict[k1][index]['te_acc'].append(_te_acc)
            acc_threshold_dict[k1][index]['th'].append(_th)

print("\n")
print("Accuracy Dense Layer: ")
for k, v in acc_dense_dict.items():
    _train_mean_acc = np.mean(v[0])
    _train_std_acc = np.std(v[0])
    _test_mean_acc = np.mean(v[1])
    _test_std_acc = np.std(v[1])

    print("\t" + k + ": train mean acc = " +
          "{:.4f}".format(_train_mean_acc) +
          ", train std = " + "{:.4f}".format(_train_std_acc) +
          "; test mean acc = " +
          "{:.4f}".format(_test_mean_acc) +
          ", test std = " +
          "{:.4f}".format(_test_std_acc))

print("\n")
print("Accuracy Bayesian Approach: ")
for k, v in acc_bayesian_dict.items():
    print(k)
    for index in range(len(v)):
        dd = v[index]
        for k2, v2 in dd.items():
            if len(v2[0]) > 0:
                _train_mean_acc = np.mean(v2[0])
                _train_std_acc = np.std(v2[0])
                _test_mean_acc = np.mean(v2[1])
                _test_std_acc = np.std(v2[1])
                print("\t" + k2 + ": train mean acc = " +
                      "{:.4f}".format(_train_mean_acc) +
                      ", train std = " + "{:.4f}".format(_train_std_acc) +
                      "; test mean acc = " +
                      "{:.4f}".format(_test_mean_acc) +
                      ", test std = " +
                      "{:.4f}".format(_test_std_acc))

print("\n")
print("Heuristic Hybrid Approach: ")
for k, v in acc_threshold_dict.items():
    print(k)
    for index in range(len(acc_threshold_dict[k])):
        _tr_acc = acc_threshold_dict[k][index]['tr_acc']
        _te_acc = acc_threshold_dict[k][index]['te_acc']
        _th = acc_threshold_dict[k][index]['th']

        _train_mean_acc = np.mean(_tr_acc)
        _train_std_acc = np.std(_tr_acc)

        _test_mean_acc = np.mean(_te_acc)
        _test_std_acc = np.std(_te_acc)

        _th_mean = np.mean(_th)
        _th_std = np.std(_th)

        print("\ttrain mean acc = " +
              "{:.4f}".format(_train_mean_acc) + ", train std acc = " +
              "{:.4f}".format(_train_std_acc) + "; test mean acc = " +
              "{:.4f}".format(_test_mean_acc) + ", test std acc = " +
              "{:.4f}".format(_test_std_acc) + "; th mean = " +
              "{:.4f}".format(_th_mean)
              + ", th std = " + "{:.4f}".format(_th_std))
