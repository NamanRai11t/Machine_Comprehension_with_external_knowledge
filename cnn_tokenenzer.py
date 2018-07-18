import json
import re
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('/home/namanrai/newsqa/stanford-corenlp-full-2018-02-27/stanford-corenlp-full-2018-02-27')

with open('cnn_dataset_combined_gold.json', 'r') as File:
    for line in File:
        temp = json.loads(line.strip())
        question = temp['question']
        gold_answer = temp['gold_answer']
        tok_question = nlp.word_tokenize(question)
        tok_story_text = nlp.word_tokenize(temp['story_text'])
        story_id = temp['story_id']
        Dict = {}
        Dict['story_id'] = story_id
        Dict['question'] = tok_question
        Dict['story_text'] = tok_story_text
        Dict['gold_answer'] = gold_answer
        print(json.dumps(Dict))
        
nlp.close()
