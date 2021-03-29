# Research code on deep learning for natural language processing/sentiment analysis
---
## The goal of this project is to explore ideas for improving the accuracy and robustness of neural networks for NLP/sentiment analysis
We draw inspiration from how humans process language. 
When people try to understand nuanced language they typically process
multiple input sensor modalities to complete this cognitive task. It turns out the
human brain has even a specialized neuron formation, called **sagittal stratum**, to
help us understand sarcasm. 

We use this biological formation as the inspiration for
designing a neural network architecture that combines predictions of different
models on the same text to construct robust, accurate and computationally
efficient classifiers for sentiment analysis and study several different realizations.
We have developed a systematic new approach to combining multiple
predictors based on a dedicated neural network. Our approach is backed by a rigorous mathematical
analysis of the combining logic along with state-of-the-art experimental results. We have also developed a
heuristic-hybrid technique for combining models that relies on a heuristic idea of separating the 
set of models into one base and the remaining auxiliary, combined with appropriate Bayesian rules.  Experimental
results on a representative benchmark dataset and comparisons to other methods show the advantages of the new approaches.

Check our research paper ["Can you tell? SSNet - a Sagittal Stratum-inspired Neural Network
  Framework for Sentiment Analysis"](https://arxiv.org/abs/2006.12958) for context and details.

| Proposed Combiner  | Max Accuracy (%) |
| ------------- | ------------- |
| Neural Network Combiner  | 94.11  |
| Bayesian Decision Rule Combiner  | 93.77  |
| Heuristic-Hybrid Combiner  | 94.0  |


## Prototype code, see the [licensing agreement](https://github.com/usnistgov/STVM_NLP_Research/blob/master/LICENSE.md) for terms of use.

## Installation Process: 
The project is written using the [tensorflow](https://www.tensorflow.org/install) library and is compatible with tensorflow 2.x. The dataset used for the experiments can be found at [SLMRD](https://ai.stanford.edu/~amaas/data/sentiment/) which consists of 25K train reviews and 25K test reviews.

## WorkFlow of SSNet
Following is the directory structure:
<img src="images/directory_structure.png" height="20">

The project consists of five modules: a) Split the dataset b) Training individual model c) Generating prediction from individual models d) Training the combiners e) Generating prediction from the combiner. Follow the below instructions to split the dataset and evaluate the models or combiners:
* Split train dataset:
	* Split IMDB labeled train data into 20K and 5K. Make sure the split data is balanced i.e, pos and pos reviews are evenly split in both files.
	* Source code: STVM_NLP_Research\model_c_d\misc-src\split-imdb-train-dataset.ipynb
	* Input: STVM_NLP_Research\model_c_d\IMDB\data\imdb_master.zip. Uncompress the file as imdb_master.csv before running.
	* Output 1: data\imdb_train_split_20000.csv
	* Output 2: data\imdb_train_split_5000.csv
* Training individual model: There are four individual models, M_1(BowTie), M_2(BLSTM), M_3(BERT) and M_4(USE)
	* Model_1: Trained on 20K dataset. 
	* Model_2: Trained on 20K dataset. 
	* Model_3: Trained on 20K dataset. 
	* Model_4: Trained on 20K dataset. 
* Generating prediction from individual models: 
	* The prediction file on 5K: "data/model_1_5ktrain.csv" and on 25K: "data/model_1_25ktest.csv"
	* The prediction file on 5K: "data/model_2_5ktrain.csv" and on 25K: "data/model_2_25ktest.csv"
	* The prediction file on 5K: "data/model_3_5ktrain.csv" and on 25K: "data/model_3_25ktest.csv"
	* The prediction file on 5K: "data/model_4_5ktrain.csv" and on 25K: "data/model_4_25ktest.csv"
* Training the combiners: please check "Combiners" section
* Generating prediction from the combiner: pleased check "Combiners" section

## Combiners
The project consists of three combiners: Neural Network Combiner, Bayesian Decision Rule Combiner and Heuristic-Hybrid Combiner. Each combiner has its own training and prediction accuracies. Following are the source codes for the combiners:
* combiners/SSNet_predictions.py: run this file for evaluating accuracy of all the three combiner proposed in the paper
* combiners/SSNet_Neural_Network.py: neural network combiner
* combiners/SSNet_Bayesian_Decision.py: bayesian decision combiner
* combiners/SSNet_Heuristic_Hybrid.py: heuristic hybrid combiner

Use following instructions to run the combiner:
1) The combiner implementation consists of four files placed in the directory "combiners". The file "SSNet_predictions.py" expects 5K train data and probabilities predicted on tain and test data by all the four models. Currently all these required files are placed inside the data folder. Please note that file names may be little different, please rename if required before next step
2) Run "SSNet_predictions.py"
3) It could take a while to complete the three proposed methods but once done the output will produce all the accuracies 
4) The output contains train (5K) accuracy and test accuracy (25K)
5) There would be minor changes in accuracy for neural network combiner

## Contacts: apostol.vassilev@nist.gov, munawar.hasan@nist.gov, honglan.jin@nist.gov



