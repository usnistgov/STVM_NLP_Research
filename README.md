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

## Prototype code, see the [licensing agreement](https://github.com/usnistgov/STVM_NLP_Research/blob/master/LICENSE.md) for terms of use.

## Installation Process: TBD

## Contacts: apostol.vassilev@nist.gov, munawar.hasan@nist.gov, honglan.jin@nist.gov



