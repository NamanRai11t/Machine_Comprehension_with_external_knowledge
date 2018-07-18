import json
import re
import pandas as pd

with open('new_output2.csv', 'r') as f:
     reader = pd.read_csv(f)
     for story in reader.values.tolist():
          story_id = story[0]
          # Make Answer list
          ans = story[2].split("|")
          ans_list = []          
          for elem in ans:
               if "," in elem:
                    split_elem = elem.split(",")
                    for elem2 in split_elem:
                         ans_list.append(elem2)
               elif ":" in elem:
                    ans_list.append(elem)     
          # Make story word list
          count = 0
          story_word_list = re.sub("\s+", " ", story[6]).split(" ")
          # # Output sample
          # print(story[6])
          # print("Question:", story[1])
          # print("Answer:", ans)
          # for elem in ans_list:
          #      split_elem = elem.split(":")
          #      print("\t", story[6][int(split_elem[0]):int(split_elem[1])])               
          # Search word index number
          answer_list = []
          for elem in ans_list:
               split_elem = elem.split(":")
               start_index = re.sub("\s+", " ", story[6][:int(split_elem[0])]).count(" ") 
               end_index = re.sub("\s+", " ", story[6][:int(split_elem[1])]).count(" ")
               answer_list.append(str(start_index)+":"+str(end_index))
               # for elem in range(start_index, end_index):
               #      print(story_word_list[elem], end=" ")
               # print()
          print(story_id+';'+str(answer_list))
