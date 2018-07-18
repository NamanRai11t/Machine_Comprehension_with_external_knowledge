#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
import numpy as np

def make_knowledge_matrix(data, obj, conceptnet_types, xp=np):
    knowledge_matrix = xp.zeros(len(data['raw_arg1'])*len(data['raw_arg2'])*len(conceptnet_types), dtype=xp.int32).reshape(len(data['raw_arg1']),len(data['raw_arg2']),len(conceptnet_types))
    seq_mask = make_seq_mask(obj)
    for label in conceptnet_types:
        if label in obj["raw_conceptnet"]:
            pair_list = obj["raw_conceptnet"][label].strip().split(" ")
            for pair in pair_list:
                start, end = pair.split("->")
                if "_" in start:
                    temp = start.split("_")
                    start = temp[-1]
                if "_" in end:
                    temp = end.split("_")
                    end = temp[-1]
                if seq_mask[int(start)] and seq_mask[int(end)] and seq_mask[int(start)] != seq_mask[int(end)]:
                    if int(end) > int(start):
                        arg1_num = seq_mask[0:int(start)+1].count('arg1') - 1
                        arg2_num = seq_mask[0:int(end)+1].count('arg2') - 1
                    else:
                        arg1_num = seq_mask[0:int(end)+1].count('arg1') - 1
                        arg2_num = seq_mask[0:int(start)+1].count('arg2') - 1
                    knowledge_matrix[arg1_num][arg2_num][conceptnet_types.index(label)] = 1
    return knowledge_matrix


def make_seq_mask(obj):
    seq_mask = []
    mask = False
    flag_skip = False
    flag_nest = False
    for word in obj['seq']:
        if re.search("arg1>", word) or re.search("arg2>",word) or re.search("dc>", word) or re.search("skip>", word):
            seq_mask.append(False)
            if word == "<skip>":
                flag_skip = True
            elif word == "</skip>":
                flag_skip = False
            elif word == "<arg1>":
                if mask == "arg2":
                    flag_nest = 2
                mask = "arg1"
            elif word == "<arg2>" or word == "<dc>":
                if mask == "arg1":
                    flag_nest = 1
                mask = "arg2"
            elif word == "</arg1>" or word == "</arg2>" or word == "</dc>":
                if flag_nest:
                    mask = "arg"+str(flag_nest)
                    flag_nest = False
                else:
                    mask = False
        else:
            if flag_skip:
                seq_mask.append(False)
            else:
                seq_mask.append(mask)
    return seq_mask


if __name__ == "__main__":
    conceptnet_types = ["coref", "Synonym", "IsA", "RelatedTo", "AtLocation", "EtymologicallyRelatedTo"]
    obj = {'type': 'Implicit', 'coref': {'24': 14, '39': 24}, 'Synonym': {'36': 13}, 'label': 'Temporal.Asynchronous', 'raw_conceptnet': {'IsA_R': '14->9 24->9 39->9 ', 'RelatedTo': '4->37 9->4 14->32 24->32 32->13 39->32 ', 'EtymologicallyRelatedTo': '4->37 ', 'Synonym': '13->36 ', 'IsA': '9->14 9->24 9->39 '}, 'IsA': {'14': 9, '24': 9, '39': 9}, 'RelatedTo': {'37': 4, '32': 24}, 'seq': ['<arg1>', 'But', 'the', 'RTC', 'also', 'requires', '``', 'working', "''", 'capital', 'to', 'maintain', 'the', 'bad', 'assets', 'of', 'thrifts', 'that', 'are', 'sold', '</arg1>', ',', 'until', 'the', 'assets', 'can', 'be', 'sold', 'separately', '.', '<arg2>', 'That', 'debt', 'would', 'be', 'paid', 'off', 'as', 'the', 'assets', 'are', 'sold', '</arg2>'], 'IsA_R': {}, 'EtymologicallyRelatedTo': {'37': 4}}
    data = {'raw_arg1': ['But', 'the', 'RTC', 'also', 'requires', '``', 'working', "''", 'capital', 'to', 'maintain', 'the', 'bad', 'assets', 'of', 'thrifts', 'that', 'are', 'sold'], 'raw_arg2': ['That', 'debt', 'would', 'be', 'paid', 'off', 'as', 'the', 'assets', 'are', 'sold']}

    knowledge_matrix = make_knowledge_matrix(data, obj, conceptnet_types)
    print(knowledge_matrix)
