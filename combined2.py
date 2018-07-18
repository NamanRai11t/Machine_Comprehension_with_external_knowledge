import csv
import sys
import pandas as pd
import json

from multiprocessing import Pool

global gold_ans_dict

def job(row):
    indices = {}
    indices["story_id"] = row[0]
    indices["question"] = row[1]
    indices["answer"] = gold_ans_dict[row[0]]
    # indices["answer"] = row[2]
    if (row[1] == "n/a" or row[1] == "nan"):
        return None
    indices["story_text"] =row[6]
    synonym_list = []
    story_words=row[story].split()  #to capture the story text
    try:
        question_words=row[question].split()  #to capture the question
    except:
        return indices
    for i, question_word in enumerate(question_words):
        for synonym in synonyms:
            if question_word in synonym:
                for sys in synonym:
                    for j, story_word in enumerate(story_words):
                        if story_word == sys:
                            synonym_list.append((i, j))
    indices["synonym"] = synonym_list
    return indices


def f(rows):    
    return [job(row) for row in rows]  # tuple of list


if __name__ == "__main__":
    with open('gold_answers', 'r') as answersFile:
        temp = []
        gold_ans_dict = {}
        for i in answersFile.readlines():
            temp=i.strip().split(';')
            gold_ans_dict[temp[0]]=temp[1]
        
    with open('output.csv', 'r') as csvFile, open('wordnet-synonyms.txt', 'r') as synonymsFile:
        global story, question
        synonyms = [set(line.strip().split(' ')) for line in synonymsFile.readlines()]
        reader = pd.read_csv(csvFile).values.tolist()
        # reader = csv.reader(csvFile)
        # header = next(reader)
        question = 1 # header.index("question")
        story = 6 # header.index("story_text")
        #cut -d , -f 1,6 combined-newsqa-data-v1.csv
        #rows = [row.strip() for row in reader]
        n_jobs = 10
        batchsize = len(reader) // n_jobs
        rows = [reader[i:i+batchsize] for i in range(0, len(reader), batchsize)]
        
        p = Pool(n_jobs)
        results = p.map(f, rows)  # list of list of list

        new_list = []
        for list in results:
            for list2 in list:
                print(json.dumps(list2))
            
