from functions import get_vector
import re
from config import model_config
import scipy
import sklearn
from math import sqrt



def weighted_add(w1, w2, alpha=0.5, beta=0.5):

    mul_w1 = alpha*w1
    mul_w2 = alpha*w2


    return mul_w1+mul_w2


def weighted_add_score(ncs, gensim_w2v_model):
    scores = []
    for nc in ncs:
        head, modifier = re.split(' ', nc)
        w1 = get_vector(head, gensim_w2v_model, model_config.input_vector_length, model_config.seed)
        w2 = get_vector(modifier, gensim_w2v_model, model_config.input_vector_length, model_config.seed)
        w_add = weighted_add(w1=w1, w2=w2)
        compound = head + '_' + modifier
        w3 = get_vector(compound, gensim_w2v_model, model_config.output_vector_length, model_config.seed)
        scores.append(cosine_similarity(w3, w_add))
    return scores



def square_rooted(x):
    return round(sqrt(sum([a*a for a in x])),3)


def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)