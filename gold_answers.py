import json
import re
import pandas as pd

with open('answers.txt', 'r') as File:
    temp = []
    for line in File.readlines():
        temp = line.strip().split(';')
        story_id = temp[0]
        answers = temp[1]
        answers = answers[1:-1]
        answers_list = re.sub("\s", "", answers).strip().split(',')
        max_count = answers_list.count(answers_list[0])
        max_value = answers_list[0]
        for elem in answers_list:
            if answers_list.count(elem) > max_count:
                max_count = answers_list.count(elem)
                max_value = elem
        print(story_id+';'+ max_value)
            
