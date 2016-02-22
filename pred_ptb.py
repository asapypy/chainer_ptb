#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import print_function
import argparse
import math
import sys
import time

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import net


parser = argparse.ArgumentParser()
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

n_epoch = 39   # number of epochs
n_units = 650  # number of units per layer
batchsize = 20   # minibatch size
bprop_len = 35   # length of truncated BPTT
grad_clip = 1    # gradient norm threshold to clip

# Prepare dataset (preliminary download dataset by ./download.py)
vocab = {}


def load_data(filename):
    global vocab, n_vocab, rdict
    words = open(filename).read().replace('\n', '<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    rdict = dict((v,k) for k,v in vocab.iteritems())
    return dataset

train_data = load_data('ptb.train.txt')
valid_data = load_data('ptb.valid.txt')
test_data = load_data('ptb.test.txt')
print('#vocab =', len(vocab))

# Prepare RNNLM model, defined in net.py
lm = net.RNNLM(len(vocab), n_units)
model = L.Classifier(lm)
model.compute_accuracy = True  # we only want the perplexity
for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.SGD(lr=1.)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

def pred(dataset):
    # prediction routine
    pred = model.copy()  # to use different state
    pred.predictor.reset_state()  # initialize state

    sum_accuracy = 0
    for i in six.moves.range(dataset.size - 1):
        x = chainer.Variable(xp.asarray(dataset[i:i + 1]), volatile='on')
        t = chainer.Variable(xp.asarray(dataset[i + 1:i + 2]), volatile='on')
        loss = pred(x, t)
        num= np.ndarray.argmax(pred.y.data)
        print(rdict[x.data[0]], rdict[t.data[0]], rdict[num])
        sum_accuracy += pred.accuracy.data
    return float(sum_accuracy) / (dataset.size - 1)


# Evaluate on test dataset
accur = pred(train_data[:100])
print('---------------------')
print('input teacher predict')
print('accuracy:', accur)
