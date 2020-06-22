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
====================================================================================================================
'''

import numpy as np
import threading
import matplotlib.pyplot as plt

model_a_prediction_file = "model_a.txt"
model_b_prediction_file = "model_b.txt"
keras_prediction_file = "labeledKerasIMDBBoW.feat"
SAMPLE_SIZE = 50000
PROBABILITY_DISCRIMINATOR_CUTOFF = 0.5

_accuracy = dict()


class Threshold:
    def __init__(self, threshold, start_bias, end_bias, bias_step, keras_prediction):
        self.threshold = threshold
        self.start_bias = start_bias
        self.end_bias = end_bias
        self.bias_step = bias_step
        self.keras_prediction = keras_prediction

    def op(self, polarity_predictions, lstm_predictions):
        print("starting for threshold", self.threshold)
        results = [[]]
        res_a = []
        res_b = []
        acc = []
        while self.start_bias <= self.end_bias:
            correct_predictions = 0
            sorted_out_predictions = 0
            wrong_predictions = 0
            unsorted_predictions = 0
            for i in range(SAMPLE_SIZE):
                _max = np.max(polarity_predictions[i])
                if _max > self.threshold:
                    predicted_sentiment = np.argmax(polarity_predictions[i])
                    if predicted_sentiment == int(self.keras_prediction[i][0]):
                        correct_predictions += 1
                    else:
                        wrong_predictions += 1
                else:
                    probability_neg = (self.start_bias * polarity_predictions[i, 0]) + \
                                      ((1.0 - self.start_bias)
                                       * lstm_predictions[i, 0])
                    probability_pos = (self.start_bias * polarity_predictions[i, 1]) + \
                                      ((1.0 - self.start_bias)
                                       * lstm_predictions[i, 1])

                    if int(self.keras_prediction[i][0]) == np.argmax([probability_neg, probability_pos]):
                        sorted_out_predictions += 1
                    else:
                        unsorted_predictions += 1

            res_a.append(self.start_bias)
            res_b.append(1.0 - self.start_bias)
            acc.append(float(correct_predictions +
                             sorted_out_predictions) / float(SAMPLE_SIZE))

            self.start_bias += self.bias_step

        results = [res_a, res_b, acc]
        _accuracy[self.threshold] = results
        print("ending for threshold", self.threshold)


def display(top=1):
    # get max for all thresholds plots
    list_of_threshold = []
    list_of_a = []
    list_of_b = []
    list_of_accuracy = []

    for k, v in _accuracy.items():
        list_of_threshold.append(k)
        a = v[0]
        b = v[1]
        acc = v[2]
        list_of_accuracy.append(max(acc))
        list_of_a.append(a[acc.index(max(acc))])
        list_of_b.append(b[acc.index(max(acc))])

    print("*******************************")
    print(list_of_threshold)
    print(list_of_a)
    print(list_of_b)
    print(list_of_accuracy)
    print("*******************************")

    # display top 'top'
    '''
    print("displaying top "+str(top)+" results")
    plots = []
    list_of_max = []
    list_of_th = []
    for k, v in _accuracy.items():
        list_of_max.append(max(v[2]))
        list_of_th.append(k)
    print(list_of_th)
    print(list_of_max)

    for i in range(top):
        index = list_of_max.index(max(list_of_max))
        plots.append(list_of_th[index])
        print("index", index)
        del list_of_max[index]
        del list_of_th[index]

    fig, axs = plt.subplots(top, figsize=(35, 20))

    counter = 0
    thresholds = "("
    for th in plots:
        print(_accuracy[th])
        a = _accuracy[th][0]
        b = _accuracy[th][1]
        acc = _accuracy[th][2]
        print("*********************")
        print(a)
        print(b)
        print(acc)
        thresholds = thresholds+str(format(th, ".2f"))+", "
        axs[counter].plot(acc)
        axs[counter].plot(acc)
        axs[counter].plot(acc.index(max(acc)), max(acc), 'ro')
        axs[counter].text(acc.index(
            max(acc)) + 1, max(
            acc), "Acc: " + str(format(max(acc), "0.5f"))
                 + "; th: " + str(format(th, ".2f"))
                 + "; (a, b): (" + str(format(a[acc.index(max(acc))], ".2f"))
                 + ", " + str(format(b[acc.index(max(acc))], ".2f")) + ")"
                 , fontsize=6)

        counter += 1
    thresholds += thresholds[:-2]+")"
    #fig.suptitle('Accuracy Plot on threshold: ' +thresholds)
    plt.show()
    '''


def main():
    base_threshold = 0.50
    cut_off_threshold = 0.99
    threshold_step = 0.02

    base_bias = 0.15
    cut_off_bias = 0.86
    bias_step = 0.01

    kf = open(keras_prediction_file, "r")
    kp = kf.read()
    kf.close()

    kp = kp.split("\n")[:-1]

    polarity_prediction = np.zeros([SAMPLE_SIZE, 2], dtype='float64')
    lstm_prediction = np.zeros([SAMPLE_SIZE, 2], dtype='float64')

    ppf = open(model_a_prediction_file, "r")
    pp = ppf.read()
    ppf.close()
    pp = pp.split("\n")

    lpf = open(model_b_prediction_file, "r")
    lp = lpf.read()
    lpf.close()
    lp = lp.split("\n")

    for i in range(SAMPLE_SIZE):
        if float(pp[i]) < PROBABILITY_DISCRIMINATOR_CUTOFF:
            polarity_prediction[i, 0] = 1.0 - float(pp[i])
            polarity_prediction[i, 1] = float(pp[i])
        elif float(pp[i]) > PROBABILITY_DISCRIMINATOR_CUTOFF:
            polarity_prediction[i, 1] = float(pp[i])
            polarity_prediction[i, 0] = 1.0 - polarity_prediction[i, 1]

        if float(lp[i]) < PROBABILITY_DISCRIMINATOR_CUTOFF:
            lstm_prediction[i, 0] = 1.0 - float(lp[i])
            lstm_prediction[i, 1] = float(lp[i])
        elif float(lp[i]) > PROBABILITY_DISCRIMINATOR_CUTOFF:
            lstm_prediction[i, 1] = float(lp[i])
            lstm_prediction[i, 0] = 1.0 - lstm_prediction[i, 1]

    list_of_th_computations = []
    list_of_threads = []
    while base_threshold <= cut_off_threshold:
        th = Threshold(threshold=base_threshold, start_bias=base_bias,
                       end_bias=cut_off_bias, bias_step=bias_step, keras_prediction=kp)
        list_of_th_computations.append(th)
        base_threshold += threshold_step

    print("Total thresholds to calculate", len(list_of_th_computations))
    for item in list_of_th_computations:
        t = threading.Thread(target=item.op, args=(
            polarity_prediction, lstm_prediction, ))
        list_of_threads.append(t)
        t.start()

    for t in list_of_threads:
        t.join()
    display(top=5)


if __name__ == "__main__":
    main()
