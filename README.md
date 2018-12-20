# Introduction

A python implementation of the model of identifyign non-compositional compounds using polynimal multivariate multiple regression presented in the EMNLP 2015 paper: [Learning Semantic Composition to Detect Non-compositionality of Multiword Expressions](https://aclweb.org/anthology/D/D15/D15-1201.pdf)



# Usage


### Concatenate compounds
In order to be able to create embeddings for training and evaluation compounds, their components have to be concatenated with an underscore 
and then replace their original form in the corpus. This can be done using `replace-compounds.py` module: 

`python preprocessing/replace-compounds.py -p2tc ./data/train_compounds.txt -p2ec ./data/eval_reddy_original_format.txt -p2corp PATH_TO_CORPUS -p2out PATH_TO_OUTPUT`

### Create word embeddings

Possibly, the most efficient way of creating word embeddings from a corpus is using `fasttext`. Runnign `fasttext` on the corpus where certain compounds are concatenated
with underscore leads to having embeddings for all words and concatenated compounds.  

`./fasttext skipgram -dim 300 -input PATH_TO_NC_REPLACED_CORPUS -output OUTPUT_PREFIX`


### Train the regression model, then rank and evaluate compounds based on their non-compositionality

In order to train your regression model, first set the respective parameters in `config/model_config.py`. Then you need to specify path to three files
as input arguments: (1) path to word embeddings created by `fasttext` (or `word2vec`). (2) path to training compounds. An example of training compounds is provided in directory `data/`. 
These noun compounds extracted from British National Corpus with a frequency of at least 10. (3) path to evaluation compounds. Currently, the script works with evaluation of 
compunds of [Reddy et al 2011](http://www.aclweb.org/anthology/I11-1024) in the original format (available [here](http://sivareddy.in/downloads#compound_noun_compositionality) and in directory 'data/'). 

Then, running `non-comp.py` script that will read the evaluation and train compounds, train a model, and evaluate the ranking with respect to human judgments provided in 
Reddy et al 2011 data, in comparison with an additive baseline. You can also print the ranked compounds by setting the `rank` argument to `true`.   

`python non-comp.py -p2vw PATH_TO_WOED_EMBEDDINGS -p2tc data/train_compounds.txt -p2ec data/eval_reddy_original_format.txt -p2out data/ -rank 'true'`

## Contact

To report bugs and other issues and if you have any question please contact: meghdad.farahmand@gmail.com