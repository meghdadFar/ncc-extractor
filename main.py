from functions import read_ncs, reddy_ncs, rank_with_score, write_score
from additive import weighted_add_score
from gensim.models.keyedvectors import KeyedVectors
from regression import noncomp_error_score, train
import logging
from config import logging_config
import scipy
import sys
import configparser

# TODO make the dimensions non parameter but extract them from input vectors


if __name__ == '__main__':


    logging.info("Reading configuration from " + sys.argv[1])
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    gensim_w2v_model = KeyedVectors.load_word2vec_format(config['PATH']['GENSIM_MODEL'], binary=False)

    train_ncs = read_ncs(config['PATH']['TRAIN_COMPUNDS'])

    predict_ncs = read_ncs(config['PATH']['PREDICT_COMPUNDS'])

    # predict_ncs = predict_ncs[0:200]

    model = train(train_ncs, predict_ncs, gensim_w2v_model, config)

    scored_ncs = noncomp_error_score(predict_ncs, gensim_w2v_model, model)

    sorted_scored_ncs = dict(sorted(scored_ncs.items(), key=lambda kv: kv[1], reverse=True))

    write_score(sorted_scored_ncs, config['PATH']['OUTPUT']+'/reg_scores_ep' + config['TRAINING']['NUM_EPOCHS'] + '.txt')

    # reg_score = regression_score(train_ncs, predict_ncs, gensim_w2v_model)

    # ranked_ncs = rank_with_score(predict_ncs, reg_score)
    
    # print(ranked_ncs)

    # write_score(ranked_ncs, reg_score, config['PATH']['OUTPUT']+'/reg_scores.txt')
    
    # eval_ncs, eval_scores = reddy_ncs(config['PATH']['PREDICTS_COMPUNDS'])

    # additive_score = weighted_add_score(eval_ncs, gensim_w2v_model)

    # write_score(eval_ncs, additive_score, args.p2out+'additive_scores.txt')

    # print('Spearman rho bet. human score and regression score', scipy.stats.spearmanr(reg_score, eval_scores))

    # print('Spearman rho bet. human score and additive score', scipy.stats.spearmanr(additive_score, eval_scores))

    # if args.rank == 'true':
    #     print('Ranking based on regression model: ', rank_with_score(eval_ncs, reg_score))
    #     print('Ranking based on additive model:  ', rank_with_score(eval_ncs, additive_score))


