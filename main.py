from functions import read_ncs, reddy_ncs, write_score
from additive import weighted_add_score
from gensim.models.keyedvectors import KeyedVectors
from regression import noncomp_error_score, train
import logging
from config import logging_config
import scipy
import sys
import configparser


if __name__ == '__main__':

    logging.info("Reading configuration from " + sys.argv[1])
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    gensim_w2v_model = KeyedVectors.load_word2vec_format(config['PATH']['GENSIM_MODEL'], binary=False)

    train_ncs = read_ncs(config['PATH']['TRAIN_COMPUNDS'])

    predict_ncs = read_ncs(config['PATH']['PREDICT_COMPUNDS'])

    predict_ncs = predict_ncs[0:1000]

    model, criterion = train(train_ncs, predict_ncs, gensim_w2v_model, config)

    scored_ncs = noncomp_error_score(predict_ncs, gensim_w2v_model, model, criterion)

    sorted_scored_ncs = dict(sorted(scored_ncs.items(), key=lambda kv: kv[1], reverse=True))

    write_score(sorted_scored_ncs, config['PATH']['OUTPUT']+'/reg_scores_ep' + config['TRAINING']['NUM_EPOCHS'] + '.txt')



