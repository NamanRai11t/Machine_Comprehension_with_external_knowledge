import json
import re
from collections import Counter
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input', '-i', default=None, type=str, dest='inpath')
args = parser.parse_args()

if args.inpath:
    with open(args.inpath, 'r') as file1:
    
        y = file1.read().split('\n')
        ulist = []
        count =0
        t_gold = 0
        t_system = 0
        for line in y[4:-4]:
            words = line.split("/ ")
            count +=1
            # y = words[1]
            # z = y.split(':')
            # t_gold += (int(z[2]) - int(z[1]))
            x = words[2]
            # z1 = x.split(':')
            # t_system += (int(z1[2]) - int(z1[1]))
            ulist.append(x)
        counts = Counter(ulist)
        print('Total ='+str(counts)+ str(len(counts)))
        #print('ave_gold_len:'+ str(t_gold/count) + 'ave_sys_len:' + str(t_system/count))


