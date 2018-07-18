from nltk.corpus import wordnet as wn
from collections import defaultdict
import json

with open('squad_dev_dataset_updated.json', 'r') as file1, open('squad_train_dataset_updated.json', 'r') as file2:
        Dict = {}
        for line in file1:

                temp = json.loads(line.strip())
                for question_word in temp['question']:
                        question_word_synsets = wn.synsets(question_word)
                        if len(question_word_synsets) > 0:
                                pickup = question_word_synsets[0:1]
                                for pickup_word in pickup:
                                        for hyper in pickup_word.hypernyms():
                                                Dict['words'] = pickup_word.lemma_names()
                                                Dict['hypernyms'] = hyper.lemma_names()
                                                print(json.dumps(Dict))
        for line in file2:
                temp = json.loads(line.strip())
                for question_word in temp['question']:
                        question_word_synsets = wn.synsets(question_word)
                        if len(question_word_synsets) > 0:
                                pickup = question_word_synsets[0:1]
                                for pickup_word in pickup:
                                        for hyper in pickup_word.hypernyms():
                                                Dict['words'] = pickup_word.lemma_names()
                                                Dict['hypernyms'] = hyper.lemma_names()
                                                print(json.dumps(Dict))

