import json
import re

with open('tok_squad_train_dataset.json','r') as File:
    story_id_pre = ''
    count = 0
    for line in File:
        start_index = 0
        end_index = 0
        temp = json.loads(line.strip())
        answer_text = temp['answer_text']
        story_text = temp['story_text']
        question = temp['question']
        story_id = temp['story_id']
        if (story_id_pre != story_id):
            count +=1
            indexes = [index for index, value in enumerate(story_text) if value == answer_text[0]]
            for index in indexes:
                if ((answer_text[0] in story_text) and (answer_text[-1] in story_text)):
                    k = index
                    l = k + len(answer_text) -1
                    if l<len(story_text):
                        if (answer_text[-1] == story_text[l]):
                            start_index = k
                            end_index = l
                            answer = str(start_index) + ':' + str(end_index)
                            Dict = {}
                            Dict['answer_text'] = answer_text
                            Dict['gold_answer'] = answer
                            Dict['story_id'] = story_id
                            Dict['question'] = question
                            Dict['story_text'] = story_text
                            print(json.dumps(Dict))
                        else:
                            Dict = {}
                            Dict['answer_text'] = answer_text
                            #Dict['gold_answer'] = answer
                            Dict['story_id'] = story_id
                            Dict['question'] = question
                            Dict['story_text'] = story_text
                            #print(json.dumps(Dict))
                            
        story_id_pre = temp['story_id']

                        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                
