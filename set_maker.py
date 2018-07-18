import json
import re
import pandas as pd



with open('cnn_dataset_combined_gold_token_updated_syn.json', 'r') as File, open('train_story_ids.csv','r') as trainIds, open('test_story_ids.csv','r') as testIds, open('dev_story_ids.csv','r') as devIds:
    train_combined = []
    test_combined = []
    reader2 = pd.read_csv(devIds).values.tolist()
    flat_list2 = [item2 for sublist2 in reader2 for item2 in sublist2]
    reader1 = pd.read_csv(testIds).values.tolist()
    flat_list1 = [item1 for sublist1 in reader1 for item1 in sublist1]
    reader = pd.read_csv(trainIds).values.tolist()
    flat_list = [item for sublist in reader for item in sublist]

    with open("cnn_train_combined_syn.json", "w") as out_train, open("cnn_test_combined_syn.json", "w") as out_test, open("cnn_dev_combined_syn.json", "w") as out_dev:
        for line in File:
            temp = json.loads(line.strip())
            story_id = temp['story_id']
            if story_id in flat_list:
                # train_combined.append(temp)
                print(json.dumps(temp), file=out_train)
            elif story_id in flat_list1:
                print(json.dumps(temp), file=out_test)
                # test_combined.append(temp)
            elif story_id in flat_list2:
                print(json.dumps(temp), file=out_dev)
                # test_combined.append(temp)
            else:
                print(temp)
                # print(json.dumps(temp), file=out_error)


