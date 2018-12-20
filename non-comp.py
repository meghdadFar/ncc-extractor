from functions import read_ncs, get_vectors, create_batch, train, get_non_comp_args, \
    get_poly_features, rank_with_score, write_score, read_eval, build_model, regression_score
from baseline import weighted_add_score
from gensim.models.keyedvectors import KeyedVectors
import logging
from config import logging_config
from config import model_config
import scipy


# TODO extract dimensions directly from input vectors and not as a parameter

if __name__ == '__main__':

    args = get_non_comp_args()

    logging.info('Reading gensim model')
    gensim_w2v_model = KeyedVectors.load_word2vec_format(args.p2vw, binary=False)

    logging.info('Reading evaluation set')
    eval_ncs, eval_scores, eval_scores_inv = read_eval(args)

    logging.info('Calculating additive score')
    additive_score = weighted_add_score(eval_ncs, gensim_w2v_model)
    write_score(eval_ncs, additive_score, args.p2out+'additive_scores.csv')

    logging.info('Reading train set')
    ncs = read_ncs(args.p2tc)

    logging.info('Creating vector for training instances')
    X, Y = get_vectors(ncs, gensim_w2v_model)
    if model_config.poly_degree > 1:
        X = get_poly_features(X, model_config.poly_degree)

    logging.info('Creating batches')
    in_batches, tar_batches = create_batch(X, Y, model_config.batch_size)

    logging.info('Creating the regression model')
    model, optimizer, criterion = build_model(X, Y)

    logging.info('Training')
    train(in_batches, tar_batches, model, model_config.nb_epochs, optimizer, criterion)

    logging.info('Calculating regression-based scores')
    reg_score = regression_score(eval_ncs, gensim_w2v_model, model)
    write_score(eval_ncs, reg_score, args.p2out+'reg_scores.csv')

    print('Spearman rho bet. inv human score and regression', scipy.stats.spearmanr(reg_score, eval_scores_inv))
    print('Spearman rho bet. inv human score and additive score', scipy.stats.spearmanr(additive_score, eval_scores_inv))

    if args.rank == 'true':
        print('Ranking based on regression model: ', rank_with_score(eval_ncs, reg_score))
        print('Ranking based on additive model:  ', rank_with_score(eval_ncs, additive_score))