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


def get_vectors(ncs, gensim_w2v_model):
    X = []
    Y = []
    for nc in ncs:
        head, modifier = re.split(' ', nc)
        w1w2 = np.append(get_vector(head, gensim_w2v_model, model_config.input_vector_length, model_config.seed),
                         get_vector(modifier, gensim_w2v_model, model_config.input_vector_length, model_config.seed))
        X.append(w1w2)
        compound = head + '_' + modifier
        y = get_vector(compound, gensim_w2v_model, model_config.output_vector_length, model_config.seed)
        Y.append(y)
    return np.array(X), np.array(Y)


def get_vector(w, gensim_w2v_model, length, seed):
    if w in gensim_w2v_model.vocab:
        return gensim_w2v_model.word_vec(w)
    else:
        logging.info('Vector not found for ' + w + '. Returning random vector')
        return random_vec(length, seed)


def train(inp_batches, tar_batches, model, num_epochs, optimizer, criterion):
    print('Training can be stopped by ctrl+c at any time. The program will continue with evaluation')
    try:
        for ep in range(0, num_epochs):
                epoch_loss = train_epoch(inp_batches, tar_batches, model, optimizer, criterion)
                logging.info('epoch '+str(ep) +'\tloss ' + str(epoch_loss))
    except KeyboardInterrupt:
        pass


def train_epoch(inp_batches, tar_batches, model, optimizer, criterion):
    avg_loss = 0
    for i in range(0, inp_batches.shape[0]):
        avg_loss += train_batch(inp_batches[i], tar_batches[i], model, optimizer, criterion)
    avg_loss = float(avg_loss/inp_batches.shape[0])
    return avg_loss


def train_batch(inp_batch, tar_batch, model, optimizer, criterion):
    inp = Variable(torch.from_numpy(inp_batch))
    tar = Variable(torch.from_numpy(tar_batch))
    mdl = model
    if use_cuda:
        mdl = mdl.cuda()
        inp = inp.cuda()
        tar = tar.cuda()

    out = mdl(inp.float())
    optimizer.zero_grad()
    loss = criterion(out.float(), tar.float())
    loss.backward()
    optimizer.step()
    return loss


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


def predict(eval_set_ncs, gensim_w2v_model, model):
    losses = []
    X, Y = get_vectors(eval_set_ncs, gensim_w2v_model)
    if model_config.poly_degree > 1:
        X = get_poly_features(X, model_config.poly_degree)
    for i in range(0, X.shape[0]):
        inp = Variable(torch.from_numpy(X[i, :]))
        tar = Variable(torch.from_numpy(Y[i, :]))
        if use_cuda:
            # TODO check of the model is not cuda throw exception
            inp = inp.cuda()
            tar = tar.cuda()
        loss = F.smooth_l1_loss(model(inp.float()), tar.float())
        losses.append(loss.data[0])
    return losses


def print_every(line, current, every):
    if current % every == 0:
        print('processing line: ' + str(current))


def random_vec(length, seed):
    np.random.seed(seed)
    vec = np.random.rand(length)
    return vec


def get_poly_features(X, degree):
    poly = PolynomialFeatures(degree, interaction_only=True)
    X2 = poly.fit_transform(X)
    return X2


def get_non_comp_args():
    parser = argparse.ArgumentParser(description="Calculates the non-compositionality of the two words noun compounds "
                                                 "using regression.")
    parser.add_argument('-p2vw', '--path-to-word-vectors', help="Path to word vectors", dest='p2vw')
    parser.add_argument('-p2tc', '--path-to-train-compounds', help="Path to training compounds", dest='p2tc')
    parser.add_argument('-p2ec', '--path-to-eval-compounds', help="Path to evaluation compounds", dest='p2ec')
    parser.add_argument('-p2out', '--path-to-out-dir', help="Path to a directory for writing results", dest='p2out')
    parser.add_argument('-rank', '--rank-evaluation-ncs', help="If true, compounds of the evaluation set will be ranked", dest='rank')
    args = parser.parse_args()
    return args


def get_preprocess_args():
    parser = argparse.ArgumentParser(description="Replace compounds in the corpus with their underscored version")
    parser.add_argument('-p2tc', '--path-to-train-compounds', help="Path to training compounds", dest='p2tc')
    parser.add_argument('-p2ec', '--path-to-eval-compounds', help="Path to evaluation compounds", dest='p2ec')
    parser.add_argument('-p2corp', '--path-to-corpus', help="Path to corpus", dest='p2corp')
    parser.add_argument('-p2out', '--path-to-out-dir', help="Path to a directory for writing results", dest='p2out')
    args = parser.parse_args()
    return args


def rank_with_score(ncs, score):
    sort_index = np.argsort(score)
    sort_index_descend = sort_index[::-1]
    ranked_ncs = [ncs[i] for i in sort_index_descend]
    return ranked_ncs


def write_score(eval_ncs, scores, path):
    outputfile = open(path, 'w')
    for i in range(0, len(eval_ncs)):
        # outputfile.write(eval_ncs[i] + ' ' + str(losses[i]) + '\n')
        outputfile.write("%s\t%.3f\n" % (eval_ncs[i], scores[i]))
    outputfile.flush()
    outputfile.close()


def read_eval(args):
    eval_ncs, eval_scores = reddy_ncs(args.p2ec)
    write_score(eval_ncs, eval_scores, args.p2out + 'eval_scores.csv')
    normalized_scores = normalize(eval_scores)
    eval_scores_inv = np.subtract(1, normalized_scores)
    write_score(eval_ncs, eval_scores_inv, args.p2out + 'eval_inv_scores.csv')
    return eval_ncs, eval_scores, eval_scores_inv


def build_model(X, Y):
    model = torch.nn.Linear((X.shape[1]), Y.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=float(model_config.learning_rate))
    criterion = F.smooth_l1_loss
    return model, optimizer, criterion


def regression_score(eval_ncs, gensim_w2v_model, regression_model):
    loss = predict(eval_ncs, gensim_w2v_model, regression_model)
    scores = np.add(loss, 0.0001)
    return scores


def normalize(data):
    data = np.array(data)
    normalized_data = (data - np.min(data)) / (np.max(data) - min(data))
    return normalized_data


def write_to_file(lines, path_to_file):
    file = open(path_to_file,'w')
    for l in lines:
        file.write(l + '\n')
    file.flush()
    file.close()
