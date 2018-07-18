import json

with open('squad_train_dataset_updated_knowledge_syn_anty_hyper.json','r') as File:
    count = 0
    c = 0
    c1=0
    c2 =0
    gold_answer1 = []
    for line in File:
        count +=1
        temp = json.loads(line.strip())
        antonym = temp['antonym']
        hypernym = temp['hypernym']
        synonym = temp['synonym']
        if (synonym != []):
            c +=1
        if (hypernym != []):
            c1 +=1
        if (antonym != []):
            c2 +=1
    print(c,c1, c2)
    print(str(c/count)+ '/'+ str(c1/count) + '/'+str(c2/count))
        
