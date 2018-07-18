import json
import re
import pandas as pd

with open('new_output.csv', 'r') as f:
    reader = pd.read_csv(f)
    for story in reader.values.tolist():
        story_id = story[0]
        question = story[1]
        story_text = story[6]
        ans = story[2].split("|")
        ans_list = []
        for elem in ans:
            if "," in elem:
                split_elem = elem.split(",")
                for elem2 in split_elem:
                    ans_list.append(elem2)
            elif ":" in elem:
                ans_list.append(elem)
        Dict = {}
        Dict['story_id'] = story_id
        Dict['question'] = question
        Dict['story_text'] = story_text
        Dict['answer'] = ans_list
        print(json.dumps(Dict))
                            
