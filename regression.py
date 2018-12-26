from config import model_config
from functions import create_batch
import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures
import re
import torch.nn.functional as F
from torch.autograd import Variable
import logging
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
        return random_vec(length, seed)


def train_epoch(inp_batches, tar_batches, model, optimizer, criterion):
    avg_loss = 0
    for i in range(0, inp_batches.shape[0]):
        Y = tar_batches[i]
        if model_config.poly_degree > 1:
            X = get_poly_features(inp_batches[i], model_config.poly_degree)
        else:
            X = inp_batches[i]
        avg_loss += train_batch(X, Y, model, optimizer, criterion)
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


def train(train_ncs, predict_ncs, gensim_w2v_model, config):
    # Prepare batches
    X, Y = get_vectors(train_ncs, gensim_w2v_model)    
    inp_batches, tar_batches = create_batch(X, Y, model_config.batch_size)

    # Set up model and optimization
    input_size = X.shape[1] if model_config.poly_degree == 1 else  get_poly_features(inp_batches[0], model_config.poly_degree).shape[1]
    output_size = Y.shape[1]
    model = torch.nn.Linear(input_size, output_size)
    optimizer = optim.SGD(model.parameters(), lr=float(config['TRAINING']['LEARNING_RATE']))
    criterion = torch.nn.SmoothL1Loss()

    # Train
    logging.info('Training can be stopped by ctrl+c at any time. The program will continue with evaluation')
    num_epochs = int(config['TRAINING']['NUM_EPOCHS'])
    try:
        for ep in range(0, num_epochs):
                epoch_loss = train_epoch(inp_batches, tar_batches, model, optimizer, criterion)
                logging.info('epoch '+str(ep) +'\tloss ' + str(epoch_loss))
    except KeyboardInterrupt:
        pass
    return model



def predict(eval_set_ncs, gensim_w2v_model, model):
    scored_ncs = {}
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
        scored_ncs[eval_set_ncs[i]] = loss.data[0]
    return scored_ncs


def noncomp_error_score(predict_ncs, gensim_w2v_model, model):
    scored_ncs = predict(predict_ncs, gensim_w2v_model, model)
    for k, v in scored_ncs.items():
        v = v[0].item()  # Get the value out of torch tensor
        # v = np.add(v, 0.001)
        # v = np.true_divide(1, v)
        scored_ncs[k] = v
    return scored_ncs