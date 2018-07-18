import json
import re
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('/home/namanrai/newsqa/stanford-corenlp-full-2018-02-27/stanford-corenlp-full-2018-02-27')

with open('cnn_dataset_combined_gold.json', 'r') as File:
    for line in File:
        temp = json.loads(line.strip())
        question = re.sub("\s+", " ", temp['question'])
        tok_question = nlp.word_tokenize(question)
        tok_answer_text = nlp.word_tokenize(temp['answer_text'])
        tok_story_text = nlp.word_tokenize(temp['story_text'])
        story_id = temp['story_id']
        answer = temp['gold_answer'].split(':')
        start_word = answer[0]
        end_word= answer[1]
        Dict = {}
        Dict['answer_text'] = tok_answer_text
        Dict['story_id'] = story_id
        Dict['question'] = tok_question
        Dict['story_text'] = tok_story_text
        Dict['start_word'] = start_word
        Dict['end_word'] = end_word
        print(json.dumps(Dict))
        
nlp.close()
