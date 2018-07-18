#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MAGE-NN model
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


def set_xp(_xp):
    global xp
    xp = _xp


class kim(Chain):
    def __init__(self, word_embed, embed_size, label_size, knowledge_size, input_hidden, kelic_hidden, enrich_hidden, mlp_hidden, input_layers=1, enrich_layers=1, dropout=0.0):
        self.word_embed = word_embed
        self.dropout = dropout
        super(kim, self).__init__()
        with self.init_scope():
            self.input_lstm_a = L.NStepBiLSTM(n_layers=input_layers, in_size=embed_size,
                                              out_size=input_hidden, dropout=self.dropout)
            self.input_lstm_b = L.NStepBiLSTM(n_layers=input_layers, in_size=embed_size,
                                              out_size=input_hidden, dropout=self.dropout)
            self.enrich_lstm_a = L.NStepBiLSTM(n_layers=enrich_layers, in_size=kelic_hidden,
                                               out_size=enrich_hidden, dropout=self.dropout)
            self.enrich_lstm_b = L.NStepBiLSTM(n_layers=enrich_layers, in_size=kelic_hidden,
                                               out_size=enrich_hidden, dropout=self.dropout)
            self.kelic_feedforward = L.Linear(input_hidden*2*4+knowledge_size, kelic_hidden)
            self.keic_feedforward = L.Linear(knowledge_size, 1)
            self.h1_to_h2 = L.Linear(enrich_hidden*2*3*2, mlp_hidden)
            self.h2_to_y = L.Linear(mlp_hidden, label_size)

    def __call__(self, a_list, b_list, a_mask, b_mask, knowledge):
        ya_ori = self.input_encoding(self.input_lstm_a , a_list)
        yb_ori = self.input_encoding(self.input_lstm_b , b_list)
        alpha, alpha_r = self.make_alpha(ya_ori, yb_ori, a_mask, knowledge)
        beta, beta_r = self.make_alpha(yb_ori, ya_ori, b_mask, xp.swapaxes(knowledge, axis1=1, axis2=2))
        ya_con, yb_con = self.kec(ya_ori, yb_ori, alpha, beta)        
        ya_loc = self.kelic(ya_ori, ya_con, alpha_r)
        yb_loc = self.kelic(yb_ori, yb_con, beta_r)
        ya_pool = self.keic(self.enrich_lstm_a, ya_loc, alpha_r)
        yb_pool = self.keic(self.enrich_lstm_b, yb_loc, beta_r)
        system_output = self.h2_to_y(F.tanh(self.h1_to_h2(F.concat((ya_pool, yb_pool), axis=1))))
        return system_output, F.concat((ya_pool, yb_pool), axis=1)
    
    def input_encoding(self, model, x_list):
        hx = None
        cx = None
        xs_f = []
        for i, x in enumerate(x_list):
            x = self.word_embed(Variable(x))
            xs_f.append(x)
        _, _, y_ori = model(hx, cx, xs_f)
        y_ori = F.stack(y_ori)
        return y_ori

    def make_alpha(self, ya, yb, mask_a, knowledge):
        exist_knowledge = xp.array(xp.sum(knowledge, axis=3) > 0, dtype=xp.int32)
        eij_matrix = F.matmul(ya, yb, transb=True) + exist_knowledge
        alpha = F.softmax(eij_matrix + (mask_a-1)[:,:,None]*float(1000000), axis=1)
        alpha_r = F.sum(F.broadcast_to(alpha[:,:,:,None], knowledge.shape) * knowledge, axis=2)
        return alpha, alpha_r
    
    def kec(self, ya_ori, yb_ori, alpha, beta):
        # Knowledge Enriched Co-attention    
        ya_con = F.stack(F.matmul(alpha, yb_ori))
        yb_con = F.stack(F.matmul(beta, ya_ori))
        return ya_con, yb_con

    def kelic(self, y_ori, y_con, alpha_r):
        # Knowledge Enriched Local Inference Collection
        batchsize, maxlen, _ = y_ori.shape
        kelic_input = F.concat((y_ori, y_con, y_ori-y_con, y_ori*y_con, alpha_r), axis=2).reshape(batchsize*maxlen, -1)
        y_loc = F.relu(self.kelic_feedforward(kelic_input)).reshape(batchsize, maxlen, -1)
        return y_loc        

    def keic(self, model, y_local, alpha_r):
        # Knowledge Enchied Inference Composition
        hx = None
        cx = None
        xs_f = []
        for i, x in enumerate(y_local):
            x = F.dropout(x, ratio=self.dropout)
            xs_f.append(x)
        _, _, y_hidden = model(hx, cx, xs_f)
        y_hidden = F.stack(y_hidden)
        # pooling
        batchsize, maxlen, embedsize = y_hidden.shape
        y_mean = F.average_pooling_nd(F.swapaxes(y_hidden, axis1=1, axis2=2), ksize=maxlen).reshape(batchsize, embedsize)
        y_max = F.max_pooling_nd(F.swapaxes(y_hidden, axis1=1, axis2=2), ksize=maxlen).reshape(batchsize, embedsize)
        weight = F.softmax(F.relu(self.keic_feedforward(alpha_r.reshape(batchsize*maxlen,-1))).reshape(batchsize, maxlen,-1), axis=1)
        y_weight = F.sum(F.broadcast_to(weight, y_hidden.shape) * y_hidden, axis=1)
        y_pooling = F.concat((y_mean, y_max, y_weight), axis=1)
        return y_pooling
        
        
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
    
    x_list = [[0, 1, 2, 3], [7, 8, -1, -1], [4, 5, 6, -1]] # 可変長データ (4, 2, 3)の長さのデータとする
    y_list = [[0, 1, -1], [2, 3, 7 ], [8, 4, 5 ]] # 可変長データ (2, 3, 3)の長さのデータとする    
    x_list = xp.array(x_list, dtype=xp.int32)
    y_list = xp.array(y_list, dtype=xp.int32)
    x_mask = (x_list != -1).astype(xp.int32)
    y_mask = (y_list != -1).astype(xp.int32)
    knowledge = [[[[1, 0],[0, 0],[0, 0]],
                  [[0, 1],[1, 0],[0, 0]],
                  [[0, 1],[0, 1],[0, 0]],
                  [[0, 1],[1, 1],[0, 0]]],
                 [[[0, 1],[0, 0],[1, 1]],
                  [[1, 0],[0, 0],[1, 1]],
                  [[0, 0],[0, 0],[0, 0]],
                  [[0, 0],[0, 0],[0, 0]]],
                 [[[1, 0],[0, 1],[1, 0]],
                  [[1, 1],[0, 1],[0, 0]],
                  [[1, 0],[0, 0],[1, 1]],
                  [[0, 0],[0, 0],[0, 0]]]]
    knowledge = xp.array(knowledge, dtype=xp.int32)
    gold = Variable(xp.array([0,2,1], dtype=xp.int32))
    
    n_vocab = 500
    emb_dim = 100
    word_embed=L.EmbedID(n_vocab, emb_dim, ignore_label=-1)
    
    use_dropout = 0.25
    label_size = 3
    knowledge_size = 2
    input_hidden = 3
    kelic_hidden = 5
    enrich_hidden = 5
    mlp_hidden = 3
    input_layers = 1
    enrich_layer = 1    

    model = kim(word_embed, emb_dim, label_size, knowledge_size, input_hidden, kelic_hidden, enrich_hidden, mlp_hidden, input_layers, enrich_layer, use_dropout)
    optimizer = optimizers.AdaGrad()
    optimizer.use_cleargrads()
    optimizer.setup(model)
    optimizer.add_hook(WeightDecay(0.0001))

    if args.gpu >= 0:
        model.to_gpu()
        word_embed.to_gpu()

    for i in range(1, args.epoch+1):
        system_output, _ = model(x_list, y_list, x_mask, y_mask, knowledge)
        loss = F.softmax_cross_entropy(system_output, gold)
        model.cleargrads()
        loss.backward()
        optimizer.update()
