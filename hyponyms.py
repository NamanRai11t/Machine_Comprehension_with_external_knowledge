from nltk.corpus import wordnet as wn
from collections import defaultdict
import json

with open('combined_new_index.txt', 'r') as file:
        for line in file:
                temp = json.loads(line.strip())
                print(temp["question"])
                question = temp["question"].split(' ')
                for question_word in question:
                        question_word_synsets = wn.synsets(question_word)
                        if len(question_word_synsets) > 0:
                                pickup = question_word_synsets[0:2]
                                for pickup_word in pickup:
                                        for hyper in pickup_word.hypernyms():
                                                print(pickup_word.lemma_names(), hyper.lemma_names())

