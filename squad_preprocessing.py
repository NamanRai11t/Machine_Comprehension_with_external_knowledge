import json

with open('train-v1.1.json', 'r') as File:
    temp = json.load(File)
    data = temp['data']
    for story in data:
        title = story['title']
        paragraphs = story['paragraphs']
        for paragraph in paragraphs:
            story_text = paragraph['context']
            qas = paragraph['qas']
            for questions in qas:
                question = questions['question']
                question_id = questions['id']
                answers = questions['answers']
                for answer in answers:
                    Dict = {}
                    start_index = answer['answer_start']
                    answer_text = answer['text']
                    Dict['start_index'] = start_word_index
                    Dict['answer_text'] = answer_text
                    Dict['story_id'] = question_id
                    Dict['question'] = question
                    Dict['story_text'] = story_text
                    print(json.dumps(Dict))




            
                
