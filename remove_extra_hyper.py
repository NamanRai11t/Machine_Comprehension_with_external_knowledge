import json

with open('check.json', 'r') as File:
    for line in File:
        temp = json.loads(line.strip())
        for line1 in temp:
            print(json.dumps(line1))
        
