#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import sys
import numpy as np
from chainer import Variable
import chainer.links as L

class WordDic(object):
    """
    Convert word into ID (Chainer Variable)
    ID=0 is reserved for UNK
    """
    offset=1
    def __init__(self, voc_size, offset=1):
        self.offset = offset
        self.k2id = {}

    def __len__(self):
        return len(self.k2id)

    def get_rawid(self, k):
        if k in self.k2id:
            return self.k2id[k]
        else:
            return 0 # UNK

    def get(self, seq, xp=np):
        if isinstance(seq, list):
            # sequence
            _id = [self.get_rawid(w) for w in seq]
            # vid = Variable(xp.array(_id, dtype=np.int32))
            vid = _id            
        else:
            # vid = Variable(xp.array([self.get_rawid(seq)], dtype=np.int32))
            vid = [self.get_rawid(seq)]
        return vid

    @classmethod
    def load_from_w2v_model(self, f, voc_limit=1000000, special_symbol_list=[]):
        line = f.readline()
        line = line.rstrip().decode('utf-8')
        voc_size, dims = line.split(" ", 1)
        if voc_limit > 0 and int(voc_size) > voc_limit:
            voc_size = voc_limit
        else:
            voc_size = int(voc_size)
        dims = int(dims)
        _self = self(voc_size)
        embed = L.EmbedID(voc_size + self.offset + len(special_symbol_list), dims, ignore_label=-1)
        binary_len = np.dtype(np.float32).itemsize * dims
        for line_no in range(voc_size):
            word = []            
            while True:
                ch = f.read(1)
                if ch == b' ':
                    break
                if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                    word.append(ch)
            word = b''.join(word)
            word = word.decode("utf-8", "ignore")
            _id = _self.k2id[word] = len(_self.k2id) + self.offset
            embed.W.data[_id] = np.fromstring(f.read(binary_len), dtype=np.float32)
        for word in special_symbol_list:
            _id = _self.k2id[word] = len(_self.k2id) + self.offset
        return _self, embed

    @classmethod
    def load_from_glove_model(self, f, voc_limit=1000000, special_symbol_list=[]):
        line = f.readline()
        line = line.rstrip()
        voc_size, dims = line.split(" ", 1)
        if voc_limit > 0 and int(voc_size) > voc_limit:
            voc_size = voc_limit
        else:
            voc_size = int(voc_size)
        dims = int(dims)
        _self = self(voc_size)
        embed = L.EmbedID(voc_size + self.offset + len(special_symbol_list), dims, ignore_label=-1)
        for line_no in range(voc_size):
            line = f.readline()
            split_line = line.strip().split(" ")
            word = split_line[0]
            _id = _self.k2id[word] = len(_self.k2id) + self.offset
            embed.W.data[_id] = np.fromstring(" ".join(split_line[1:]), dtype=np.float32, sep=' ')
        for word in special_symbol_list:
            _id = _self.k2id[word] = len(_self.k2id) + self.offset
        return _self, embed


class PWordDic(WordDic):
    """
    Convert word into ID (Chainer Variable)
    model contains <UNK> for unknown words
    """
    offset=0
    def __init__(self, voc_size, offset=0):
        self.offset = offset
        self.k2id = {}

    def get_rawid(self, k):
        if k in self.k2id:
            return self.k2id[k]
        else:
            if "<UNK>" in self.k2id:
                return self.k2id["<UNK>"]
            else:
                # print("UNK:", k)
                return 0 # UNK

if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        wdic, embed = PWordDic.load_from_w2v_model(f, voc_limit=1000000)
    _id = wdic.get("be")
    _id = Variable(np.array(_id, dtype=np.int32))
    print(_id)    
    print(embed(_id))

