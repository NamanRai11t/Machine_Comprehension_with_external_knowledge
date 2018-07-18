import json
import re
from collections import Counter


with open('oov_list.txt', 'r') as File:
    y = File.read().split('\n')
    ulist = []
    for line in y[1:-3]:
        words = line.split(':')
        ulist.append(words[1])
    counts = Counter(ulist)
    print('Total ='+str(counts))
                 
            
