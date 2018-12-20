from functions import read_ncs, reddy_ncs, get_non_comp_args, rank_with_score, write_score
from additive import weighted_add_score
from gensim.models.keyedvectors import KeyedVectors
from regression import regression_score
import logging
from config import logging_config
import scipy


# TODO make the dimensions non parameter but extract them from input vectors


if __name__ == '__main__':

    args = get_non_comp_args()

    gensim_w2v_model = KeyedVectors.load_word2vec_format(args.p2vw, binary=False)

    ncs = read_ncs(args.p2tc)

    eval_ncs, eval_scores = reddy_ncs(args.p2ec)

    additive_score = weighted_add_score(eval_ncs, gensim_w2v_model)

    write_score(eval_ncs, additive_score, args.p2out+'additive_scores.txt')

    reg_score = regression_score(ncs, eval_ncs, gensim_w2v_model)

    write_score(eval_ncs, reg_score, args.p2out+'reg_scores.txt')

    print('Spearman rho bet. human score and regression score', scipy.stats.spearmanr(reg_score, eval_scores))

    print('Spearman rho bet. human score and additive score', scipy.stats.spearmanr(additive_score, eval_scores))

    if args.rank == 'true':
        print('Ranking based on regression model: ', rank_with_score(eval_ncs, reg_score))
        print('Ranking based on additive model:  ', rank_with_score(eval_ncs, additive_score))


