import csv
import sys
import pandas as pd

from multiprocessing import Pool


def job(row):
	indices = []
	story_words=row[story].split()  #to capture the story text
	question_words=row[question].split()  #to capture the question 
	for i, question_word in enumerate(question_words):
		for synonym in synonyms:
			if question_word in synonym:
				for sys in synonym:
					for j, story_word in enumerate(story_words):
						indices.append((i, j))
	return indices


def f(rows):
	return [job(row) for row in rows]  # tuple of list


with open('output.csv', 'r') as csvFile, open('wordnet-synonyms.txt', 'r') as synonymsFile:
	synonyms = [set(line.split(',')) for line in synonymsFile.readlines()]
	global story, question
	reader = pd.read_csv(csvFile).values.tolist()
	# reader = csv.reader(csvFile)
	# header = next(reader)
	question=1 # header.index("question")
	story=6 # header.index("story_text")
	#cut -d , -f 1,6 combined-newsqa-data-v1.csv
	#rows = [row.strip() for row in reader]
	n_jobs = 10
	batchsize = len(reader) // n_jobs
	rows = [reader[i:i+batchsize] for i in range(0, len(reader), batchsize)]


	p = Pool(n_jobs + 1)
	results = p.map(f, rows)  # list of list of list

	print(results)


csvFile.close()