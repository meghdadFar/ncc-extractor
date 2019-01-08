# Introduction

A python implementation of the model of identifyign non-compositional compounds using the polynimal regression presented in the EMNLP 2015 paper: [Learning Semantic Composition to Detect Non-compositionality of Multiword Expressions](https://aclweb.org/anthology/D/D15/D15-1201.pdf)



# Usage


### Concatenate compounds
In order to be able to create embeddings for the training (and to be identified non-compositional) compounds, their constituents have to be concatenated with an underscore and then their concatenated version must replace their original form in the corpus. This can be done using `replace-compounds.py` module: 

`python preprocessing/replace-compounds.py -p2tc ./data/train_compounds.txt -p2ec ./data/eval_reddy_original_format.txt -p2corp PATH_TO_CORPUS -p2out PATH_TO_OUTPUT`

### Create word embeddings

Possibly, the most efficient way of creating word embeddings from a corpus is using `fasttext`. Runnign `fasttext` on the corpus where certain compounds are concatenated results in both single word and compound embeddings.  

`./fasttext skipgram -dim 100 -input PATH_TO_NC_REPLACED_CORPUS -output OUTPUT_PREFIX`


### Train the regression model, then rank the compounds based on their non-compositionality

In order to train your regression model, first set the respective meta parameters in `config/config.ini`. You should specify path to two files: 

(1) path to word embeddings created by `fasttext` 
(2) path to training compounds. An example of training compounds is provided in directory `data/`. 

Then, run `main.py` that will read the train and to-be-predicted compounds, train a model, and score the to-be-scored compounds with respect to their non-compositionality.

`python main.py config/config.ini`

## Contact

To report bugs and other issues and if you have any question please contact: meghdad.farahmand@gmail.com
