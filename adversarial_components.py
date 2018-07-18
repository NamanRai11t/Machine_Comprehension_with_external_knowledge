#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import sys
from argparse import ArgumentParser

import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, Variable
from chainer import optimizers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.optimizer import WeightDecay


class discriminator(Chain):
    def __init__(self, hdims, dropout=0.0):
        self.dropout = dropout
        super(adversarial, self).__init__()
        with self.init_scope():
            self.highway = L.Highway(hdims)
            self.h_to_y = L.Linear(hdims, 2)

    def __call__(self, h):
        h_highway = self.highway(Variable(h))
        y_adv = self.h_to_y(h_highway)
        return y_adv
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int, dest="gpu",
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=25, type=int, dest="epoch",
                        help='Number of epoch')
    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = np
    
    h_list = [[0.07693971, 0.24652258, 0.36516414, 0.32531304, 0.80600521,
               0.48166672, 0.28477982, 0.79841319, 0.13048268, 0.31438035],
              [0.36202242, 0.05815253, 0.24430117, 0.75336765, 0.53273125,
               0.90717045, 0.26057194, 0.17050579, 0.26934674, 0.69990416],
              [0.77141326, 0.23113243, 0.02778885, 0.35061881, 0.50881063,
               0.2445026 , 0.87910554, 0.58546545, 0.59878369, 0.03310525],
              [0.862771  , 0.54340754, 0.79409784, 0.94202909, 0.12964679,
               0.34659084, 0.18709705, 0.32934376, 0.69122394, 0.65063928],
              [0.60503851, 0.44435167, 0.96214351, 0.28996983, 0.99434833,
               0.89644887, 0.11432005, 0.65593003, 0.15861048, 0.51386829]]
    gold = Variable(xp.array([0,0,1,1,1], dtype=xp.int32))

    discriminator = discriminator(10, dropout=0.0)
    optimizer = optimizers.AdaGrad()
    optimizer.use_cleargrads()
    optimizer.setup(discriminator)
    optimizer.add_hook(WeightDecay(0.0001))

    for i in range(1, args.epoch+1):
        system_output = discriminator(h)
        loss = F.softmax_cross_entropy(system_output, gold)
        adversarial.cleargrads()
        loss.backward()
        optimizer.update()
        print(loss, F.argmax(system_output, axis=1))
