import csv
import sys
import pandas as pd
import json

from multiprocessing import Pool

def job(row):
    indices = {}
    indices["story_id"] = row['story_id']
    indices["question"] = row['question']
    indices["answer_text"] = row['answer_text']
    indices["gold_answer"] = row['gold_answer']
    indices["synonym"] = row['synonym']
    indices["antonym"] = row['antonym']
    indices["story_text"] = row['story_text']
    hypernyms_list = []
    with open('wordnet_hypernyms.json', 'r') as hypernymFile:
        for line in hypernymFile:
            temp = json.loads(line.strip())
            words = temp['words']
            hypernyms = temp['hypernyms']
            for i, question_word in enumerate(row['question']):
                if question_word in words:
                    for j, story_word in enumerate(row['story_text']):
                        if story_word in hypernyms:
                            hypernyms_list.append((i, j))
    indices["hypernym"] = hypernyms_list
    return indices


def f(rows):    
    return [job(row) for row in rows]  # tuple of list

            
with open('squad_train_dataset_updated_knowledge_syn_anty.json', 'r') as File:
    reader = []
    for line in File:
        temp = json.loads(line.strip())
        reader.append(temp)
    n_jobs = 10
    batchsize = len(reader) // n_jobs
    rows = [reader[i:i+batchsize] for i in range(0, len(reader), batchsize)]
    
    p = Pool(n_jobs)
    results = p.map(f, rows)  # list of list of list
    
    new_list = []
    for list in results:
        for list2 in list:
            if list2:
                print(json.dumps(list2))
            
