import logging
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import PolynomialFeatures
import codecs
import argparse
from config import model_config
import torch.optim as optim



use_cuda = torch.cuda.is_available()


def read_ncs(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        lines = list(line for line in lines if is_valid(line))
    return lines


def is_valid(input_str):
    mcl = model_config.min_constituent_len
    if input_str:
        w1, w2 = input_str.split(' ')
        if len(w1) > mcl and len(w2) > mcl:
            return True
    else:
        return False


def reddy_ncs(file_path):
    ncs = []
    scores = []
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            # res = re.search('(\w+)-\w+\s(\w+)-\w+\s', l)
            res = re.match('(\w+)-\w+\s(\w+)-\w+\t([\s0-9.]+)$', l)
            if res:
                nc = res.group(1) + ' ' + res.group(2)
                ncs.append(nc)
                scores.append(float(res.group(3).split(" ")[4]))
    return ncs, scores


def read_corpus(file_path):
    with codecs.open(file_path, "r", encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        lines = list(line for line in lines if line)
    return lines


def pre_process(sentence, lemmatizer):
    s = sentence.lower()
    tokens = tokenize(s)
    s = lemmatize(tokens, lemmatizer)
    s = ' '.join(s)
    s = s.strip()
    return s


def tokenize(sentence):
    tokens = word_tokenize(sentence)
    return tokens


def lemmatize(tokens, lemmatizer):
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return lemmas


def replace_compounds(sentence, compounds):
    for cmpd in compounds:
        if cmpd in sentence:
            sentence = re.sub(r'\b%s\b' % cmpd, cmpd.replace(' ', '_'), sentence)
    return sentence


def create_batch(inp, tar, batch_size):
    if not match_size(inp, tar):
        logging.error('Size of input and target must match')
        return None
    inp_batches = []
    tar_batches = []
    for i in range(0, inp.shape[0], batch_size):
        inp_batches.append(inp[i:i+batch_size, :])
        tar_batches.append(tar[i:i+batch_size, :])
    return np.array(inp_batches), np.array(tar_batches)


def match_size(X, Y):
    if X.shape[0] == Y.shape[0]:
        return True
    else:
        return False



def print_every(line, current, every):
    if current % every == 0:
        print('processing line: ' + str(current))


def write_score(scored_ncs, path):
    outputfile = open(path, 'w')
    for k, v in scored_ncs.items():
        outputfile.write("%s\t%.3f\n" % (k, v))
    outputfile.flush()
    outputfile.close()


def read_eval(args):
    eval_ncs, eval_scores = reddy_ncs(args.p2ec)
    write_score(eval_ncs, eval_scores, args.p2out + 'eval_scores.csv')
    normalized_scores = normalize(eval_scores)
    eval_scores_inv = np.subtract(1, normalized_scores)
    write_score(eval_ncs, eval_scores_inv, args.p2out + 'eval_inv_scores.csv')
    return eval_ncs, eval_scores, eval_scores_inv


def write_to_file(lines, path_to_file):
    file = open(path_to_file,'w')
    for l in lines:
        file.write(l + '\n')
    file.flush()
    file.close()
