#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
from collections import defaultdict, Counter

def make_knowledge_matrix(obj, wordnet_types, xp=np):
    knowledge_matrix = xp.zeros(len(obj['question'])*len(obj['story_text'])*len(wordnet_types), dtype=xp.int32).reshape(len(obj['question']),len(obj['story_text']),len(wordnet_types))
    for label in wordnet_types:
        if label == "None":
            continue
        for (question, story) in obj[label]:
            knowledge_matrix[question][story][wordnet_types.index(label)] = 1            
    return knowledge_matrix


if __name__ == "__main__":
    wordnet_types = ["synonym"]
    obj = {'answer': '34:60|1610:1618|34:60', 'story_id': './cnn/stories/c48228a52f26aca65c31fad273e66164f047f292.story', 'V_answer': "nan", 'synonym': [(2, 231), (2, 263), (2, 231), (2, 263), (2, 231), (2, 263), (2, 231), (2, 263), (2, 231), (2, 263), (2, 231), (2, 263), (2, 231), (2, 263), (2, 231), (2, 263), (2, 231), (2, 263), (2, 231), (2, 263), (2, 231), (2, 263), (2, 231), (2, 263), (2, 231), (2, 263), (2, 231), (2, 263), (2, 231), (2, 263), (2, 231), (2, 263), (2, 231), (2, 263), (3, 23)], 'question': "Where was one employee killed?", 'story_text':"(CNN) -- Fighting in the volatile Sudanese region of Darfur has sparked another wave of refugees into Chad and left a Red Cross employee dead, according to international agencies.Refugee camps in eastern Chad house about 300,000 people who fled violence in the Darfur region of Sudan.The U.N. High Commissioner for Refugees said on Monday that more than 12,000 people have fled militia attacks over the last few days from Sudan's Darfur region to neighboring Chad, still recovering from a recent attempt by rebels there to topple the government.\"\"Most of the new arrivals in Chad had already been displaced in Darfur in recent years. They are really tired of being attacked and having to move,\"\" said UNHCR's Jorge Holly. \"\"All the new refugees we talked to said they did not want to go back to Darfur at this point, they wanted to be transferred to a refugee camp in eastern Chad.\"\"This latest influx of refugees in Chad aggravates an already deteriorating security situation across this politically unstable region of Africa.Before the latest flight into Chad, the UNHCR and its partner groups \"\"were taking care of 240,000 Sudanese refugees in 12 camps in eastern Chad and some 50,000 from Central African Republic in the south of the country.\"\" Up to 30,000 people in Chad fled the country for Cameroon during the rebel-government fighting.The International Committee of the Red Cross said on Monday that one of its employees was killed in western Darfur last week during fighting. The victim is a 45-year-old Sudanese national and father of six children.He was killed in the area of Seleia, one of the three towns where reported government-backed Janjaweed militia attacks on Friday left around 200 people dead.U.N. Secretary-General Ban Ki-moon last week deplored the acts, urged all parties to stop hostilities, and said \"\"all parties must adhere to international humanitarian law, which prohibits military attacks against civilians.\"\"The United Nations says \"\"more than 200,000 people have been killed and 2.2 million others forced to flee their homes since fighting began in 2003 among government forces, rebel groups and allied militia groups known as the Janjaweed \"\"The recent fight between Chad's government and rebels is seen as a proxy war over Darfur. Sudan's government believes Chad is supporting rebels in Darfur. Chad's government believes Sudan is sup porting the rebels that moved on Chad's capital of N'Djamena. E-mail to a friend"}
    obj["question"] = obj["question"].split(" ")
    obj["story_text"] = obj["story_text"].split(" ")    
    knowledge_matrix = make_knowledge_matrix(obj, wordnet_types)
    print(knowledge_matrix)
