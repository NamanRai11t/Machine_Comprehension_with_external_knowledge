import json
import re

with open('cnn_dataset_combined.json', 'r') as File:
    for line in File:
        temp = json.loads(line.strip())
        story_id = temp['story_id']
        story_text = re.sub("\s+"," ",temp['story_text'])
        answers = temp['answer']
        question = temp['question']
        story_list = story_text.split(" ")
        if (answers != [] and question != ''):
            max_count = answers.count(answers[0])
            max_value = answers[0]
            for elem in answers:
                if answers.count(elem) > max_count:
                    max_count = answers.count(elem)
                    max_value = elem
            elm = max_value.split(':')
            start_index = story_text[:int(elm[0])].count(' ')
            end_index = story_text[:int(elm[1])].count(' ')
            gold_answer = str(start_index-1) + ':' + str(end_index-1)
            answer_text = " ".join(story_list[start_index-1:end_index])
            Dict = {}
            Dict['story_id'] = story_id
            Dict['question'] = question
            Dict['story_text'] = story_text
            Dict['gold_answer'] = gold_answer
            Dict['answer_text'] = answer_text
            print(json.dumps(Dict))
                                                            
                                                                                            
