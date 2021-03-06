from functions import create_batch
import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures
import re
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import tqdm
from config import logging_config

use_cuda = torch.cuda.is_available()

def random_vec(length, seed):
    np.random.seed(seed)
    vec = np.random.rand(length)
    return vec


def get_poly_features(X, degree):
    poly = PolynomialFeatures(degree, interaction_only=True)
    X2 = poly.fit_transform(X)
    return X2


def get_vectors(ncs, gensim_w2v_model, config):
    X = []
    Y = []
    for nc in ncs:
        head, modifier = re.split(' ', nc)
        w1w2 = np.append(get_vector(head, gensim_w2v_model, int(config['GENERAL']['INPUT_VEC_LEN']), int(config['GENERAL']['SEED'])),
                         get_vector(modifier, gensim_w2v_model, int(config['GENERAL']['INPUT_VEC_LEN']), int(config['GENERAL']['SEED'])))
        X.append(w1w2)
        compound = head + '_' + modifier
        y = get_vector(compound, gensim_w2v_model, int(config['GENERAL']['OUTPUT_VEC_LEN']), int(config['GENERAL']['SEED']))
        Y.append(y)
    return np.array(X), np.array(Y)


def get_vector(w, gensim_w2v_model, length, seed):
    if w in gensim_w2v_model.vocab:
        return gensim_w2v_model.word_vec(w)
    else:
        return random_vec(length, seed)


def train_epoch(inp_batches, tar_batches, model, optimizer, criterion, config):
    avg_loss = 0
    logging.info('Training batches')
    for i in tqdm.tqdm(range(0, inp_batches.shape[0])):
        Y = tar_batches[i]
        if int(config['GENERAL']['POLY_DEGREE']) > 1:
            X = get_poly_features(inp_batches[i], int(config['GENERAL']['POLY_DEGREE']))
        else:
            X = inp_batches[i]
        avg_loss += train_batch(X, Y, model, optimizer, criterion)
    avg_loss = float(avg_loss/inp_batches.shape[0])
    return avg_loss


def train_batch(inp_batch, tar_batch, model, optimizer, criterion):
    inp = Variable(torch.from_numpy(inp_batch))
    tar = Variable(torch.from_numpy(tar_batch))
    if use_cuda:
        inp = inp.cuda()
        tar = tar.cuda()
    out = model(inp.float())
    optimizer.zero_grad()
    loss = criterion(out.float(), tar.float())
    loss.backward()
    optimizer.step()
    return loss


def train(train_ncs, predict_ncs, gensim_w2v_model, config):
    # Prepare batches
    X, Y = get_vectors(train_ncs, gensim_w2v_model, config)    
    inp_batches, tar_batches = create_batch(X, Y, int(config['TRAINING']['BATCH_SIZE']))

    # Set up model and optimization
    input_size = X.shape[1] if int(config['GENERAL']['POLY_DEGREE']) == 1 else  get_poly_features(inp_batches[0], int(config['GENERAL']['POLY_DEGREE'])).shape[1]
    output_size = Y.shape[1]
    model = torch.nn.Linear(input_size, output_size)
    if use_cuda:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=float(config['TRAINING']['LEARNING_RATE']))
    criterion = torch.nn.SmoothL1Loss()

    # Train
    logging.info('Training can be stopped by ctrl+c at any time. The program will continue with evaluation')
    num_epochs = int(config['TRAINING']['NUM_EPOCHS'])
    try:
        for ep in range(0, num_epochs):
                epoch_loss = train_epoch(inp_batches, tar_batches, model, optimizer, criterion, config)
                logging.info('epoch '+str(ep) +'\tloss ' + str(epoch_loss))
    except KeyboardInterrupt:
        pass
    return model, criterion



def predict(ncs, gensim_w2v_model, model, config):
    scored_ncs = {}
    X, Y = get_vectors(ncs, gensim_w2v_model, config)

    logging.info('Scoring compounds')

    for nc in tqdm.tqdm(ncs):
        head, modifier = re.split(' ', nc)
        w1w2 = np.append(get_vector(head, gensim_w2v_model, int(config['GENERAL']['INPUT_VEC_LEN']), int(config['GENERAL']['SEED'])),
                         get_vector(modifier, gensim_w2v_model, int(config['GENERAL']['INPUT_VEC_LEN']), int(config['GENERAL']['SEED'])))
        compound = head + '_' + modifier
        y = get_vector(compound, gensim_w2v_model, int(config['GENERAL']['OUTPUT_VEC_LEN']), int(config['GENERAL']['SEED']))

        y = y.reshape(1, -1)
        w1w2 = w1w2.reshape(1, -1)

        inp = get_poly_features(w1w2, int(config['GENERAL']['POLY_DEGREE'])) if int(config['GENERAL']['POLY_DEGREE']) > 1 else w1w2

        inp = Variable(torch.from_numpy(np.array(inp)))
        tar = Variable(torch.from_numpy(y))

        # inp = Variable(torch.from_numpy(X[i, :]))
        # tar = Variable(torch.from_numpy(Y[i, :]))

        if use_cuda:
            # TODO check of the model is not cuda throw exception
            inp = inp.cuda()
            tar = tar.cuda()
        loss = F.smooth_l1_loss(model(inp.float()), tar.float())
        scored_ncs[nc] = loss.item() 
    return scored_ncs




def predict_batch(ncs, gensim_w2v_model, model, criterion, config):

    # Prepare batches
    output_ncs = []
    output_scores = []
    start_index = 0

    X, Y = get_vectors(ncs, gensim_w2v_model, config)    
    inp_batches, tar_batches = create_batch(X, Y, int(config['TRAINING']['BATCH_SIZE']))

    criterion = torch.nn.SmoothL1Loss(reduce=False, size_average=True)
    logging.info('Scoring batches')
    for i in tqdm.tqdm(range(0, inp_batches.shape[0])):
        Y = tar_batches[i]
        if int(config['GENERAL']['POLY_DEGREE']) > 1:
            X = get_poly_features(inp_batches[i], int(config['GENERAL']['POLY_DEGREE']))
        else:
            X = inp_batches[i]
        
        inp = Variable(torch.from_numpy(X))
        tar = Variable(torch.from_numpy(Y))

        if use_cuda:
            inp = inp.cuda()
            tar = tar.cuda()
        
        out = model(inp.float())
        loss = criterion(out.float(), tar.float())
        end_index = start_index + int(config['TRAINING']['BATCH_SIZE']) 

        output_ncs.extend(ncs[start_index:end_index])
        output_scores.extend(loss.tolist())
        start_index += int(config['TRAINING']['BATCH_SIZE'])

    return output_ncs, output_scores


def noncomp_error_score(predict_ncs, gensim_w2v_model, model, criterion, config):
    scored_ncs = predict(predict_ncs, gensim_w2v_model, model, config)
    return scored_ncs