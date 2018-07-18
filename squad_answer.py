import json
import re

with open('tok_squad_dev_dataset.json','r') as File:
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
        start_word_index = temp['start_char']
        len_answer = len(answer_text)
        index_list = []
        if (story_id_pre != story_id):
            count +=1 
            indexes = [index for index, value in enumerate(story_text) if value == answer_text[0]]
            if ((answer_text[0] in story_text) and (answer_text[-1] in story_text)):
                for index in indexes:
                    c = 0
                    for i in range(len_answer):
                        if((index+len_answer-1)<len(story_text)):
                            if (answer_text[i]==story_text[index+i]):
                                c +=1
                            else:
                                break
                    if (c==len_answer):
                        index_list.append(index)
                if (len(index_list)>0):
                    k = min(index_list, key=lambda x:abs(x-int(start_word_index)))
                    l = k + len(answer_text) -1
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
                    Dict['gold_answer'] = '1000:1005'
                    Dict['story_id'] = story_id
                    Dict['question'] = question
                    Dict['story_text'] = story_text
                    print(json.dumps(Dict))
                    
            else:
                Dict = {}
                Dict['answer_text'] = answer_text
                Dict['gold_answer'] = '1000:1005'
                Dict['story_id'] = story_id
                Dict['question'] = question
                Dict['story_text'] = story_text
                print(json.dumps(Dict))

        story_id_pre = temp['story_id']


        
        
    
