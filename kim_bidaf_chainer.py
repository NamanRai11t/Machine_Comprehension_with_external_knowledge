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
from chainer import Function, Variable, variable
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
            self.modeling_start_lstm = L.NStepBiLSTM(n_layers=enrich_layers, in_size=kelic_hidden,
                                                     out_size=enrich_hidden, dropout=self.dropout)
            self.modeling_end_lstm = L.NStepBiLSTM(n_layers=enrich_layers, in_size=enrich_hidden*2,
                                                   out_size=enrich_hidden, dropout=self.dropout)
            self.kelic_feedforward = L.Linear(input_hidden*2*4+knowledge_size, kelic_hidden)
            self.W1 = variable.Parameter(xp.random.rand(kelic_hidden+enrich_hidden*2).reshape(1,-1))
            self.W2 = variable.Parameter(xp.random.rand(kelic_hidden+enrich_hidden*2).reshape(1,-1))            
            
    def __call__(self, a_list, b_list, a_mask, b_mask, knowledge):
        # a_list: Question
        # b_list: Story text
        ya_ori = self.input_encoding(self.input_lstm_a , a_list, a_mask)
        yb_ori = self.input_encoding(self.input_lstm_b , b_list, b_mask)
        alpha, _ = self.make_alpha(ya_ori, yb_ori, a_mask, knowledge) # (minibatch, maxlen(a_list), maxlen(b_list))
        beta, beta_r = self.make_alpha(yb_ori, ya_ori, b_mask, xp.swapaxes(knowledge, axis1=1, axis2=2)) # (minibatch, maxlen(b_list), maxlen(a_list))
        ya_con, yb_con = self.kec(ya_ori, yb_ori, alpha, beta)
        # ya_loc = self.kelic(ya_ori, ya_con)
        yb_loc = self.kelic(yb_ori, yb_con, beta_r)
        h_start = self.modeling(self.modeling_start_lstm, yb_loc, b_mask)
        h_end = self.modeling(self.modeling_end_lstm, h_start, b_mask)
        batchsize, _, hidden_size = F.concat((yb_loc, h_start), axis=2).shape
        system_start = F.matmul(F.broadcast_to(self.W1, (batchsize,1,hidden_size)), F.concat((yb_loc, h_start), axis=2), transb=True).reshape(batchsize,-1)
        system_end = F.matmul(F.broadcast_to(self.W2, (batchsize,1,hidden_size)), F.concat((yb_loc, h_end), axis=2), transb=True).reshape(batchsize,-1)
        return system_start, system_end
    
    def input_encoding(self, model, x_list, x_mask):
        hx = None
        cx = None
        xs_f = []
        for i, x in enumerate(x_list):
            x = self.word_embed(Variable(x))
            xs_f.append(x)
        _, _, y_ori = model(hx, cx, xs_f)        
        y_ori = F.stack(self.add_padding(y_ori, x_mask))
        return y_ori

    def make_alpha(self, ya, yb, mask_a, knowledge):
        exist_knowledge = xp.array(xp.sum(knowledge, axis=3) > 0, dtype=xp.int32)
        eij_matrix = F.matmul(ya, yb, transb=True) + exist_knowledge
        alpha = F.softmax(eij_matrix + (mask_a-1)[:,:,None]*float(1000000), axis=1)
        alpha_r = F.sum(F.broadcast_to(alpha[:,:,:,None], knowledge.shape) * knowledge, axis=2)
        return alpha, alpha_r
    
    def kec(self, ya_ori, yb_ori, alpha, beta):
        # Knowledge Enriched Co-attention
        ya_con = F.matmul(alpha, yb_ori)
        yb_con = F.matmul(beta, ya_ori)
        return ya_con, yb_con

    def kelic(self, y_ori, y_con, alpha_r):
        # Knowledge Enriched Local Inference Collection
        batchsize, maxlen, _ = y_ori.shape
        kelic_input = F.concat((y_ori, y_con, y_ori-y_con, y_ori*y_con, alpha_r), axis=2).reshape(batchsize*maxlen, -1)
        y_loc = F.relu(self.kelic_feedforward(kelic_input)).reshape(batchsize, maxlen, -1)
        return y_loc        

    def modeling(self, model, x_list, x_mask):
        # Knowledge Enchied Inference Composition
        hx = None
        cx = None
        xs_f = []
        for i, x in enumerate(x_list):
            x = F.get_item(x,(slice(0,x_mask[i].sum()),)) # Remove padding
            x = F.dropout(x, ratio=self.dropout)
            xs_f.append(x)            
        _, _, y_hidden = model(hx, cx, xs_f)
        y_hidden = F.stack(self.add_padding(y_hidden, x_mask))
        return y_hidden

    def add_padding(self, data, mask):
        _, maxlen = mask.shape
        output = []
        for y in data:
            pad = maxlen - y.shape[0]
            if pad > 0:
                y = F.vstack((y, Variable(self.xp.zeros((pad, y.shape[1]), dtype=xp.float32))))
            output.append(y)
        return output

        
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

    x_list = [[0, 1], [2, 3, 7], [8, 4, 5]] # 可変長データ (2, 3, 3)の長さのデータとする    
    y_list = [[0, 1, 2, 3], [7, 8], [4, 5, 6]] # 可変長データ (4, 2, 3)の長さのデータとする
    x_list = [np.array(x, dtype=np.int32) for x in x_list]
    y_list = [np.array(y, dtype=np.int32) for y in y_list]
    x_mask = [[1, 1, 0], [1, 1, 1], [1, 1, 1]]
    y_mask = [[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]]    
    x_mask = xp.array(x_mask, dtype=xp.int32)
    y_mask = xp.array(y_mask, dtype=xp.int32)    

    knowledge = [[[[1,0],[0,0],[1,0],[0,1]],
                  [[0,1],[1,0],[0,0],[1,1]],
                  [[0,0],[0,0],[0,0],[0,0]]],
                 [[[0,1],[0,0],[0,0],[0,0]],
                  [[1,0],[0,0],[0,0],[0,0]],
                  [[0,0],[1,1],[0,0],[0,0]]],
                 [[[1,0],[0,1],[1,0],[0,0]],
                  [[1,1],[0,1],[0,0],[0,0]],
                  [[0,0],[0,0],[1,1],[0,0]]]]
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
        system_start, system_end = model(x_list, y_list, x_mask, y_mask, knowledge)
        loss = F.softmax_cross_entropy(system_start, gold)
        print(F.argmax(system_start, axis=1), "loss:", loss.data)
        model.cleargrads()
        loss.backward()
        optimizer.update()
