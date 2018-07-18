#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from collections import defaultdict

class analyze():
    def __init__(self, LOG=sys.stdout):
        self.LOG = LOG

    def prc_result(self, result_dict):
        Total_TP = 0
        Total_FP = 0
        Total_FN = 0
        F_list = {}
        for key in sorted(result_dict):
            if(key == "Gold" or key == "System"): continue
            TP = result_dict[key]["TP"]
            FP = result_dict[key]["FP"]
            FN = result_dict[key]["FN"]
            F_list[key] = self.prc_calc(key,TP,FP,FN)
            if("談話関係なし" not in key):
                Total_TP += result_dict[key]["TP"]
                Total_FP += result_dict[key]["FP"]
                Total_FN += result_dict[key]["FN"]
        F_list["MicroAve"] = self.prc_calc("MicroAve", Total_TP, Total_FP, Total_FN)
        return F_list["MicroAve"], F_list
                
    def prc_calc(self, key, TP, FP, FN):
        P = float(TP) / (TP + FP) if TP + FP != 0 else 0
        R = float(TP) / (TP + FN) if TP + FN != 0 else 0
        F = float(2*TP) / (2*TP + FP + FN) if 2*TP + FP + FN != 0 else 0
        print(key, "=>", file=self.LOG, end="")
        print("Precision:%.4f (%d/%d) " % (P, TP, TP + FP), file=self.LOG, end="")
        print("Recall:%.4f (%d/%d) " % (R, TP, TP + FN), file=self.LOG, end="")
        print("F-measure:%.4f" % (F), file=self.LOG)
        return F
            
    def count_up(self, result_dict, gold, system):    
        if gold == system:
            result_dict[system]["TP"] += 1
        else:
            result_dict[system]["FP"] += 1
            result_dict[gold]["FN"] += 1
            
    def read_result_log(self,filename, result_dict):
        for line in open(filename):
            # Gold\tSystem\tSoftMax
            temp = line.strip().split("\t")
            self.count_up(result_dict, temp[0],temp[1])

            
if __name__ == "__main__":
    result_dict = defaultdict(lambda: defaultdict(int))
    analyze = analyze()
    analyze.read_result_log(sys.argv[1], result_dict)
    analyze.prc_result(result_dict)
