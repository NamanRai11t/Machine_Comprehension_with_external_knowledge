#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import copy
import random
import json
from collections import defaultdict, Counter
from argparse import ArgumentParser

import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, Variable
from chainer import optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.optimizer import WeightDecay

from kim_chainer import kim, set_xp
from adversarial_components import discriminator
from embed import PWordDic
from make_knowledge_matrix import make_knowledge_matrix
from conceptnet_labeling import conceptnet_labellist
from analyze import analyze

def make_train_data(f, wdic, label_dic, conceptnet_types, xp=np):
    train_list = []
    unk_count, total = 0, 0
    unk_freq = Counter()
    for line in f.readlines():
        obj = json.loads(line.strip())
        data = {}
        # remove words before <arg*>
        flag_arg1 = False
        flag_arg2 = False
        flag_skip = False
        flag_nest = False
        data["raw_arg1"] = []
        data["raw_arg2"] = []
        for i, word in enumerate(obj["seq"]):
            if flag_arg1:
                if word == "</arg1>":
                    flag_arg1 = False
                    flag_nest = False
                elif word == "<skip>":
                    flag_skip = True
                elif word == "</skip>":
                    flag_skip = False
                elif not flag_skip:
                    if word == "<arg2>" or word == "<dc>":
                        flag_nest = 1
                    elif word == "</arg2>" or word == "</dc>":
                        flag_nest = False
                    elif not flag_nest == 1:
                        data["raw_arg1"].append(word)                    
            if flag_arg2:
                if word == "</arg2>" or word == "</dc>":
                    flag_arg2 = False
                    flag_nest = False
                elif word == "<skip>":
                    flag_skip = True
                elif word == "</skip>":
                    flag_skip = False
                elif not flag_skip:
                    if word == "<arg1>":
                        flag_nest = 2
                    elif word == "</arg1>":
                        flag_nest = False
                    elif not flag_nest == 2:
                        data["raw_arg2"].append(word)
            if not flag_arg1 and (word == "<arg1>"):
                flag_arg1 = True
            elif not flag_arg2 and (word == "<arg2>" or word == "<dc>"):
                flag_arg2 = True
        data["dat_arg1"] = wdic.get(data["raw_arg1"], xp=xp)
        data["dat_arg2"] = wdic.get(data["raw_arg2"], xp=xp)
        total += len(data["dat_arg1"])
        total += len(data["dat_arg2"])        
        unk_count += data["dat_arg1"].count(0)
        unk_count += data["dat_arg2"].count(0)
        for word, wid in zip(data["raw_arg1"], data["dat_arg1"]):
            if wid == 0:
                unk_freq[word] += 1
        for word, wid in zip(data["raw_arg2"], data["dat_arg2"]):
            if wid == 0:
                unk_freq[word] += 1
        data["link"] = make_knowledge_matrix(data, obj, conceptnet_types, xp).tolist()
        data["label"] = obj["label"]
        data["dat_label"] = label_dic[obj["label"]]
        train_list.append(data)
    print("unk rate {} ({} / {})".format(float(unk_count) / total, unk_count, total))
    for i, word in enumerate(sorted(unk_freq.keys(), key=lambda x: unk_freq[x], reverse=True)):
        if i >= 10: break
        print("<unk>\t{}\t{}".format(word, unk_freq[word]))
    return train_list


def batch_make(data_list, size=16, shuffle=False, xp=np):
    x_list, y_list, x_mask, y_mask, knowledge, gold = [], [], [], [], [], []
    select = ""
    if shuffle:
        print("shuffling...")
        random.shuffle(data_list)
    for data in data_list:
        x_list.append(data["dat_arg1"])
        y_list.append(data["dat_arg2"])
        knowledge.append(data["link"])
        gold.append(data["dat_label"])        
        if len(x_list) >= size:
            x_list, x_mask = convert_Variable(x_list, xp)
            y_list, y_mask = convert_Variable(y_list, xp)        
            knowledge = padding_to_knowledge_matrix(knowledge, len(x_list[0]), len(y_list[0]), xp)
            yield x_list, y_list, x_mask, y_mask, knowledge, xp.array(gold, dtype=xp.int32), select
            x_list, y_list, x_mask, y_mask, knowledge, gold = [], [], [], [], [], []
            select = ""
    if len(x_list) > 0:
        x_list, x_mask = convert_Variable(x_list, xp)
        y_list, y_mask = convert_Variable(y_list, xp)
        knowledge = padding_to_knowledge_matrix(knowledge, len(x_list[0]), len(y_list[0]), xp)
        yield x_list, y_list, x_mask, y_mask, knowledge, xp.array(gold, dtype=xp.int32), select


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


def padding_to_knowledge_matrix(knowledge, x_maxlen, y_maxlen, xp=np):
    output_knowledge = copy.deepcopy(knowledge)
    pad = [0] * len(output_knowledge[0][0][0])
    for k in range(0,len(output_knowledge)):
        for i in range(0, len(output_knowledge[k])):
            for j in range(len(output_knowledge[k][i]), y_maxlen):
                output_knowledge[k][i].append(pad)
        for i in range(len(output_knowledge[k]), x_maxlen):
            output_knowledge[k].append([pad] * y_maxlen)
    return xp.array(output_knowledge, dtype=xp.int32)
    
    
def load_label_list(f):
    label_dic = {}
    for line in f:
        line = line.rstrip()
        label_dic[line] = len(label_dic)
    return label_dic


def make_conceptnet_types(filename, conceptnet=False, coref=False):
    make_labellist = conceptnet_labellist()
    if conceptnet:
        raw_types = []
        with open(filename, 'r') as f:
            for line in f:
                raw_types.append(line.strip())
        # Concidering the direction of each labels
        symmetric_relations = make_labellist.def_symmetric_relations()
        conceptnet_types = make_labellist.make_fulllist(raw_types, symmetric_relations)
    else:
        conceptnet_types = []
    if coref:
        conceptnet_types.append("coref")
    return conceptnet_types


class testing():
    def __init__(self, label_dic, LOG=sys.stdout):
        self.rev_label_dic={v:k for k,v in label_dic.items()}
        self.analyze = analyze(LOG)
        self.best_f_score = 0

    def __call__(self, model, test_data, update=False, test_log=False, xp=np):
        result_dict = defaultdict(lambda: defaultdict(int))
        update_flag = False
        for x_list, y_list, x_mask, y_mask, knowledge, gold, select in batch_make(test_data, size=1, shuffle=False, xp=xp):
            system_output, _ = model(x_list, y_list, x_mask, y_mask, knowledge)
            for y in system_output:
                max_score = 0
                max_number = 0
                for score in y.data.tolist():
                    if max_score < score:
                        max_score = score
                        max_number = y.data.tolist().index(score)
            self.analyze.count_up(result_dict, self.rev_label_dic[int(gold)], self.rev_label_dic[max_number])
            if test_log:
                sys.stdout.write("%(label)s\t%(max)s\t%(raw)s\n" % {'label':self.rev_label_dic[int(gold)], 'max':self.rev_label_dic[max_number], 'raw':json.dumps(y.data.tolist())})
        f_score, f_list = self.analyze.prc_result(result_dict)
        if update and f_score > self.best_f_score:
            print('Update:', self.best_f_score, '->', f_score)
            self.best_f_score = f_score
            update_flag = True
        return f_score, f_list, update_flag


def set_optimizer(model):
    optimizer = optimizers.AdaGrad()
    optimizer.use_cleargrads()
    optimizer.setup(model)
    optimizer.add_hook(WeightDecay(0.0001))
    return optimizer


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int, dest="gpu",
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=25, type=int, dest="epoch",
                        help='Number of epoch')
    parser.add_argument('--config', '-c', default="config", type=str, dest="config",
                        help='Config file')
    parser.add_argument('--model', '-m', default="kim", type=str, dest="model",
                        help="Model name")
    parser.add_argument('--test_only', '-t', default=False, action="store_true",
                        dest="test_only")
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
    with open(config['dataset']['labellist'], "r") as f:
        label_dic = load_label_list(f)
    with open(config['dataset']['word2vec'], "rb") as f:
        wdic, word_embed = PWordDic.load_from_w2v_model(f, voc_limit=int(config['parameter']['vocab_size']))
    conceptnet_types = make_conceptnet_types(filename=config['dataset']['conceptnet_list'], conceptnet=True)
    if not args.test_only:
        with open(config['dataset']['inpath'], "r") as f:
            train_data = make_train_data(f, wdic, label_dic, conceptnet_types, xp)
        with open(config['dataset']['devpath'], "r") as f:
            dev_data = make_train_data(f, wdic, label_dic, conceptnet_types, xp)
    
    # model setting
    kim_imp = kim(word_embed, int(config['parameter']['emb_dim']), len(label_dic), len(conceptnet_types),
                  int(config['hiddenlayer']['input_hidden']), int(config['hiddenlayer']['kelic_hidden']),
                  int(config['hiddenlayer']['enrich_hidden']), int(config['hiddenlayer']['mlp_hidden']),
                  int(config['other']['input_layer']), int(config['other']['enrich_layer']), 0.0)
    optimizer_imp = set_optimizer(kim_imp)
    if args.model == "full":
        kim_exp = kim(word_embed, int(config['parameter']['emb_dim']), len(label_dic), len(conceptnet_types),
                      int(config['hiddenlayer']['input_hidden']), int(config['hiddenlayer']['kelic_hidden']),
                      int(config['hiddenlayer']['enrich_hidden']), int(config['hiddenlayer']['mlp_hidden']),
                      int(config['other']['input_layer']), int(config['other']['enrich_layer']), 0.0)
        discriminator = discriminator(int(config['other']['enrich_layer'])*3*2, 0.0)
        optimizer_exp = optimizers.AdaGrad(kim_exp)
        optimizer_dis = optimizers.AdaGrad(adversarial)       
    save_name = config['dataset']['outdir'] + '/' + config['dataset']['name']
    best_name = config['dataset']['outdir'] + '/best_' + config['dataset']['name']
    dev_test = testing(label_dic)
    
    if args.gpu >= 0:
        kim_imp.to_gpu()        
        word_embed.to_gpu()
        if args.model == "full":
            kim_exp.to_gpu()
            discriminator.to_gpu()

    if not args.test_only:
        # training
        print("Training start...")
        for i in range(1, args.epoch+1):
            print("Epoch:", i)
            for x_list, y_list, x_mask, y_mask, knowledge, gold, select in batch_make(train_data, size=int(config['parameter']['batch_size']), shuffle=True, xp=xp):
                if args.model == "kim":
                    system_output, _ = kim_imp(x_list, y_list, x_mask, y_mask, knowledge)
                    loss = F.softmax_cross_entropy(system_output, gold)
                    kim_imp.cleargrads()
                    loss.backward()
                    optimizer_imp.update()
                    print("train loss:", loss.data)
                elif args.model == "full":
                    if select == "Implicit":
                        system_output, hidden_state = kim_imp(x_list, y_list, x_mask, y_mask, knowledge)
                        loss_cls = F.softmax_cross_entropy(system_output, gold)
                        kim_imp.cleargrads()
                        loss_cls.backward()
                        optimizer_imp.update()
                        gold_adv = Variable(xp.zeros(batch_size, dtype=xp.int32))
                    else:
                        system_output, hidden_state = kim_exp(x_list, y_list, x_mask, y_mask, knowledge)
                        loss_cls = F.softmax_cross_entropy(system_output, gold)
                        kim_exp.cleargrads()
                        loss_cls.backward()
                        optimizer_exp.update()
                        gold_adv = Variable(xp.ones(batch_size, dtype=xp.int32))
                    y_adv = discriminator(hidden_state.data)
                    loss_adv = F.softmax_cross_entropy(y_adv, gold_adv)
                    adversarial.cleargrads()
                    loss_adv.backward()
                    optimizer_adv.update()
                    print("train loss:", loss_cls.data, "/", loss_adv.data)
            # Develop
            _, _, update_flag = dev_test(kim_imp, dev_data, update=True, xp=xp)
            if update_flag:
                serializers.save_npz(best_name + "_kim_imp.npz" , kim_imp)
                if args.model == "full":
                    serializers.save_npz(best_name + "_kim_exp.npz" , kim_exp)
                    serializers.save_npz(best_name + "_disc.npz" , discriminator)            
        # Model Save
        serializers.save_npz(save_name + "_kim_imp.npz" , kim_imp)
        if args.model == "full":
            serializers.save_npz(save_name + "_kim_exp.npz" , kim_exp)
            serializers.save_npz(save_name + "_disc.npz" , discriminator)
        
    #Testing
    serializers.load_npz(best_name + "_kim_imp.npz", kim_imp)
    with open(config['dataset']['testpath'], "r") as f:
        test_data = make_train_data(f, wdic, label_dic, conceptnet_types, xp)
    print("Test start... -> model:", best_name, " TestData:", config['dataset']['testpath'])
    _, _, _, = dev_test(kim_imp, test_data, xp=xp)
