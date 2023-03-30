import os
import sys
import nltk

#  python bootstrap_sentences.py coha/$year/all_$year.txt
path = sys.argv[1]
i = sys.argv[2]

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


sentences = []
with open(path) as f:
	for line in f:
		line = line.replace("@", "")
		split = tokenizer.tokenize(line)
		sentences.extend(split)

from random import choices

print (len(sentences))
year = path.split("_")[-1].split(".")[0]
with open("coha/" + year + "/bootstrapped" + str(i) + "_" + year + ".txt", "w") as f:
	for line in choices(sentences, k=len(sentences)):
		f.write(line + "\n")

