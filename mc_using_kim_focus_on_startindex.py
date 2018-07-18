#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import codecs
import copy
import random
import re
import json
import pickle
import itertools
from collections import defaultdict, Counter
from argparse import ArgumentParser

import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.optimizer import WeightDecay
from chainer import initializers
from chainer.cuda import to_cpu

from kim_chainer_without_knowledge import kim, set_xp
# from kim_chainer import kim, set_xp
from embed import PWordDic
from knowledge_matrix import make_knowledge_matrix
#from conceptnet_labeling import conceptnet_labellist

def make_train_data(f, wdic, conceptnet_types, maxlen=None, xp=np):
    train_list = []
    print("limitation of max length in story text:", maxlen)
    discard = 0
    unk_count = 0
    total_word = 0
    for line in f.readlines():
        obj = json.loads(line.strip())
        data = {}
        if obj["gold_answer"] == "":
            continue
        elif maxlen and int(maxlen) < len(obj["story_text"].split(" ")):
            discard += 1
            continue
        # data["link"] = make_knowledge_matrix(obj, conceptnet_types, xp).tolist()
        data["link"] = []
        data["answer"] = obj["gold_answer"]
        data["story_text"] = obj["story_text"].split(" ")
        data["question"] = obj["question"].split(" ")
        data["dat_question"] = wdic.get(data["question"], xp=xp)
        data["dat_story_text"] = wdic.get(data["story_text"], xp=xp)
        unk_count += data["dat_story_text"].count(0)
        total_word += len(data["dat_story_text"])
        train_list.append(data)
    print("The number of the discard stories:", discard)
    print("UNK word rate:", float(unk_count) / total_word, "(", unk_count, "/", total_word, ")")
    return train_list


def batch_make(data_list, size=16, shuffle=False, xp=np):
    x_list, y_list, x_mask, y_mask, knowledge, gold_start, gold_end, question = [], [], [], [], [], [], [], []
    if shuffle:
        print("shuffling...")
        random.shuffle(data_list)
    for data in data_list:
        x_list.append(data["dat_question"])
        y_list.append(data["dat_story_text"])
        knowledge.append(data["link"])
        split_gold = re.sub("'", "", data["answer"]).split(":")
        gold_start.append(int(split_gold[0]))
        gold_end.append(int(split_gold[1]))
        question.append(data["question"])
        if len(x_list) >= size:
            x_list, x_mask = convert_Variable(x_list, xp)
            y_list, y_mask = convert_Variable(y_list, xp)
            # knowledge = padding_to_knowledge_matrix(knowledge, len(x_list[0]), len(y_list[0]), xp)
            knowledge = []
            yield x_list, y_list, x_mask, y_mask, knowledge, xp.array(gold_start, dtype=xp.int32), xp.array(gold_end, dtype=xp.int32), question
            x_list, y_list, x_mask, y_mask, knowledge, gold_start, gold_end, question = [], [], [], [], [], [], [], []
    if len(x_list) > 0:
        x_list, x_mask = convert_Variable(x_list, xp)
        y_list, y_mask = convert_Variable(y_list, xp)
        # knowledge = padding_to_knowledge_matrix(knowledge, len(x_list[0]), len(y_list[0]), xp)
        knowledge = []
        yield x_list, y_list, x_mask, y_mask, knowledge, xp.array(gold_start, dtype=xp.int32), xp.array(gold_end, dtype=xp.int32), question


def convert_Variable(list_data, xp=np):
    max_len = 0
    for elem in list_data:
        if max_len < len(elem):
            max_len = len(elem)
    output_list = copy.deepcopy(list_data)
    for elem in output_list:
        pad = [-1] * (max_len - len(elem))
        elem.extend(pad)
    output_list = xp.array(output_list, dtype=xp.int32)
    output_mask = (output_list != -1).astype(xp.int32)
    return output_list, output_mask


# def padding_to_knowledge_matrix(knowledge, x_maxlen, y_maxlen, xp=np):
#     output_knowledge = copy.deepcopy(knowledge)
#     pad = [0] * len(output_knowledge[0][0][0])
#     for k in range(0,len(output_knowledge)):
#         for i in range(0, len(output_knowledge[k])):
#             for j in range(len(output_knowledge[k][i]), y_maxlen):
#                 output_knowledge[k][i].append(pad)
#         for i in range(len(output_knowledge[k]), x_maxlen):
#             output_knowledge[k].append([pad] * y_maxlen)
#     return xp.array(output_knowledge, dtype=xp.int32)
    
    
def make_conceptnet_types(filename):
    conceptnet_types = []
    with open(filename, 'r') as f:
        for line in f:
            conceptnet_types.append(line.strip())
    return conceptnet_types


class testing():
    def __init__(self, LOG=sys.stdout):
        self.best_acc = 0
        self.best_f_score = 0

    def __call__(self, newsqa, test_data, update=False, test_log=False, xp=np):
        result_list = []
        update_flag = False
        for x_list, y_list, x_mask, y_mask, knowledge, gold_start, gold_end, question in batch_make(test_data, size=16, shuffle=False, xp=xp):
            system_start, system_end, _ = newsqa(x_list, y_list, x_mask, y_mask, knowledge, gold_start, gold_end)
            result_list.extend(self.judge_true(system_start, system_end, gold_start, gold_end))
            if test_log:
                start_index = system_start.data.argmax(axis=1)
                # We only consider the softmax after start index number.
                end_index = []
                for i in range(0,len(start_index.data)):
                    max_score = -10000
                    max_number = start_index[i] + 1
                    for score in system_end[i].data.tolist()[max_number:]:
                        if max_score < score:
                            max_score = score
                            max_number = system_end[i].data.tolist().index(score)
                    end_index.append(max_number)
                for i in range(0, len(question)):
                    print("[Question]", " ".join(question[i]), " / Gold:", gold_start[i], ":", gold_end[i], " / System:", start_index[i], ":", end_index[i], " / ", system_start.data[i].tolist())
        acc = float(result_list.count(True))/len(result_list)
        f_score = 0
        print(acc, "(", int(result_list.count(True)), "/", len(result_list))
        # f_score, f_list = self.analyze.prc_result(result_dict)
        if update and acc > self.best_acc:
            print('Update:', self.best_acc, '->', acc)
            self.best_acc = acc
            update_flag = True
        return acc, f_score, update_flag

    def judge_true(self, system_start, system_end, gold_start, gold_end):
        system_start.to_cpu()
        system_end.to_cpu()
        gold_start = to_cpu(gold_start)
        # gold_end = to_cpu(gold_end)
        start_index = system_start.data.argmax(axis=1)
        # end_index = system_end.data.argmax(axis=1)
        judge_list = []
        for i in range(0, len(start_index)):
            # if int(start_index.data[i]) >= int(gold_start.data[i]) and int(end_index.data[i]) <= int(gold_end.data[i]) and int(start_index.data[i]) < int(end_index.data[i]):
            if int(start_index.data[i]) == int(gold_start.data[i]):
                judge_list.append(True)
            else:
                judge_list.append(False)
        return judge_list

    
def set_optimizer(model):
    optimizer = optimizers.AdaGrad()
    optimizer.use_cleargrads()
    optimizer.setup(model)
    optimizer.add_hook(WeightDecay(0.0001))
    return optimizer


class newsqa(Chain):
    def __init__(self, word_embed, config, conceptnet_types, softmax_length):
        self.dropout = float(config['other']['use_dropout'])
        self.softmax_length = int(softmax_length)
        super(newsqa, self).__init__()
        with self.init_scope():
            # self.end_LSTM = L.NStepBiLSTM(n_layers=1, in_size=int(softmax_length),
            #                               out_size=int(softmax_length), dropout=float(config['other']['use_dropout']))
            self.start_to_h = L.Linear(int(softmax_length), int(config['hiddenlayer']['end_hidden']))
            self.h_to_end = L.Linear(int(config['hiddenlayer']['end_hidden']), int(softmax_length))
            self.model = kim(word_embed, int(config['parameter']['emb_dim']), int(softmax_length), len(conceptnet_types),
                             int(config['hiddenlayer']['input_hidden']), int(config['hiddenlayer']['kelic_hidden']),
                             int(config['hiddenlayer']['enrich_hidden']), int(config['hiddenlayer']['mlp_hidden']),
                             int(config['other']['input_layer']), int(config['other']['enrich_layer']), float(config['other']['use_dropout']))

    def __call__(self, x_list, y_list, x_mask, y_mask, knowledge, gold_start, gold_end):
        system_start, _ = self.model(x_list, y_list, x_mask, y_mask, knowledge)
        loss_start = F.softmax_cross_entropy(system_start, gold_start)
        h = F.relu(self.start_to_h(system_start))
        system_end = self.h_to_end(h)
        return system_start, system_end, loss_start
        # loss_end = F.softmax_cross_entropy(system_end, gold_end)
        # losses = loss_start + loss_end
        # return system_start, system_end, losses
        

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int, dest="gpu",
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=25, type=int, dest="epoch",
                        help='Number of epoch')
    parser.add_argument('--config', '-c', default="config", type=str, dest="config",
                        help='Config file')
    parser.add_argument('--test_only', '-t', default=False, action="store_true",
                        dest="test_only")
    parser.add_argument('--load_npz', '-l', default=None, type=str, dest="npz",
                        help="Model name (if needed)")
    parser.add_argument('--optimizer', '-o', default=None, type=str, dest="optimizer",
                        help="Optimizer name (if needed)")
    parser.add_argument('--best_dev_score', '-f', default=None, type=str, dest="best_dev",
                        help="Best score in dev set (if needed)")
    parser.add_argument('--start_epoch_num', '-s', default=0, type=int, dest="start_epoch",
                        help='Start number (if needed)')
    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = np
    set_xp(xp)

    # load a config file
    import configparser
    config = configparser.ConfigParser()
    config.read(args.config)

    # load data
    with open(config['dataset']['word2vec'], "rb") as f:
        wdic, word_embed = PWordDic.load_from_w2v_model(f, voc_limit=int(config['parameter']['vocab_size']))

    conceptnet_types = []
    if config['dataset']['conceptnet_list'] != "":
        conceptnet_types = make_conceptnet_types(filename=config['dataset']['conceptnet_list'])
        
    if not args.test_only:
        with open(config['dataset']['inpath'], "r") as f:
            train_data = make_train_data(f, wdic, conceptnet_types, config['parameter']['limit_of_maxlen'], xp)
        with open(config['dataset']['devpath'], "r") as f:
            dev_data = make_train_data(f, wdic, conceptnet_types, None, xp)
    
    # model setting
    newsqa = newsqa(word_embed, config, conceptnet_types, config['parameter']['softmax_length'])
    optimizer = set_optimizer(newsqa)
    save_name = config['dataset']['outdir'] + '/' + config['dataset']['name']
    best_name = config['dataset']['outdir'] + '/best_' + config['dataset']['name']
    dev_test = testing()

    if args.gpu >= 0:
        newsqa.to_gpu()
        word_embed.to_gpu()

    # load models (if needed)
    if args.npz:
        serializers.load_npz(args.npz, newsqa)
    if args.best_dev:
        with open(args.best_dev, mode='rb') as f:
            dev_test.best_acc = pickle.load(f)
    if args.optimizer:
        serializers.load_npz(args.optimizer, optimizer)        

    if args.test_only:
        #Testing
        serializers.load_npz(best_name + ".npz", newsqa)
        with open(config['dataset']['testpath'], "r") as f:
            test_data = make_train_data(f, wdic, conceptnet_types, None, xp)
        print("Test start... -> model:", best_name, " TestData:", config['dataset']['testpath'])
        _, _, _, = dev_test(newsqa, test_data, test_log=True, xp=xp)
    else:
        # training
        print("Training start...")
        for i in range(1, args.epoch+1):
            print("Epoch:", args.start_epoch + i)
            for x_list, y_list, x_mask, y_mask, knowledge, gold_start, gold_end, _ in batch_make(train_data, size=int(config['parameter']['batch_size']), shuffle=True, xp=xp):
                _, _, loss = newsqa(x_list, y_list, x_mask, y_mask, knowledge, gold_start, gold_end)                
                newsqa.cleargrads()
                loss.backward()
                optimizer.update()
                print("train loss:", loss.data)
            # Develop & Model Save
            _, _, update_flag = dev_test(newsqa, dev_data, update=True, xp=xp)
            serializers.save_npz(save_name + "_epoch" + str(args.start_epoch + i) + ".npz" , newsqa)
            if update_flag:
                serializers.save_npz(best_name + ".npz" , newsqa)
        serializers.save_npz(save_name + "_optimizer_epoch" + str(args.start_epoch + i) + ".npz" , optimizer)
        with open(save_name + "_best_dev_score_in_epoch" + str(args.start_epoch + i) + '.pickle', mode='wb') as f:
            pickle.dump(dev_test.best_acc, f)
