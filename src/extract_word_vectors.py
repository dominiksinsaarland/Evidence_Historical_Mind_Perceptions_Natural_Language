import os
import pandas as pd
import json
import fasttext
from gensim.models import Word2Vec
from gensim.models import KeyedVectors



def load_data(experiment):
	if experiment == "our_survey":
		df = pd.read_csv("data/items-agg.csv")
		return df.item.tolist()

	elif experiment == "ggw":
		df = pd.read_csv("data/gray_wegner_data.csv")
		entities, agency, patiency = df["Entity"].tolist(), df["GGW_Agency"].tolist(), df["GGW_Experience"].tolist()
		personhood = [i + j for i,j in zip(agency, patiency)]
		entities[9] = "pvs-patient"
		entities[2] = "deceased person"
		return entities

	elif experiment == "corporate_insecthood":
		df = pd.read_csv("data/corporate_insecthood_data.csv")
		# only non-corporate entities
		df = df[(df["Category"] == "Artifact / object") | (df["Category"] == "Non-human life") | (df["Category"] == "Human")]
		entities = [e.lower() for e in df["Entity"]]
		return entities

def extract_vectors(entities, outfile_name, mode="commoncrawl"):
	"""
	small helper function, load all word vectors and extract the ones we need,
	speeds up further experiments, because we don't need to always wait for 2 minutes for the vectors to be loaded
	"""

	vectors = {}
	entities = [i.lower() for i in set(entities)]


	if mode == "glove-commoncrawl":
		# glove embeddings trained on common crawl: http://nlp.stanford.edu/data/glove.840B.300d.zip
		# Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)
		#with open("/home/dominsta/Documents/agency_experience/word-embeddings/glove.840B.300d.txt") as f:
		with open("word-vectors/glove.840B.300d.txt") as f:
			for i, line in enumerate(f):
				if i % 10000 == 0:
					print (i)
				line = line.strip().split()
				if line[0] in entities:
					vectors[line[0].lower()] = list(map(float, line[1:]))

	elif mode == "gigaword-300d":
		import gensim.downloader as api
		word_vectors = api.load("glove-wiki-gigaword-300")
		for entity in entities:
			if entity.lower() in word_vectors:
				vectors[entity.lower()] = word_vectors[entity.lower()].tolist()
			elif entity in word_vectors:
				vectors[entity.lower()] = word_vectors[entity].tolist()
			else:
				print (entity)

	elif mode == "google-newsvectors":
		# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
		from gensim import models
		path = "../appendix_paper/word-vectors/GoogleNews-vectors-negative300.bin"
		word_vectors = models.KeyedVectors.load_word2vec_format(path, binary=True)
		"""
		for entity in entities:
			if entity.lower() in word_vectors:
				vectors[entity.lower()] = word_vectors[entity.lower()].tolist()
			elif entity in word_vectors:
				vectors[entity.lower()] = word_vectors[entity].tolist()
			else:
				print (entity)
		"""
		for entity in entities:
			if entity in word_vectors:
				vectors[entity] = word_vectors[entity].tolist()
	elif mode == "fasttext-1mio-news":
		import io
		fname = "word-vectors/wiki-news-300d-1M.vec"
		with io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
			for line in f:
				line = line.strip().split()
				if line[0] in entities:
					vectors[line[0].lower()] = list(map(float, line[1:]))

	elif mode == "fasttext-commoncrawl":
		ft = fasttext.load_model('word-vectors/cc.en.300.bin')
		vectors = {i: ft.get_word_vector(i).tolist() for i in entities}
	elif mode == "fasttext-commoncrawl-subwords":
		ft = fasttext.load_model("word-vectors/crawl-300d-2M-subword.bin")
		vectors = {i: ft.get_word_vector(i).tolist() for i in entities}

	elif mode == "glove-coha":
		path_coha = "models-coha-2000s/glove.model"
		model = KeyedVectors.load(path_coha)
		vectors = {i:model[i].tolist() for i in entities if i in model}
	elif mode == "word2vec-coha":
		path_coha = "models-coha-2000s/word2vec.model"
		model = Word2Vec.load(path_coha)
		vectors = {i:model.wv[i].tolist() for i in entities if i in model.wv}

	elif mode == "fasttext-coha":
		path_coha = "models-coha-2000s/fasttext.bin"
		model = fasttext.load_model(path_coha)
		vectors = {i: model.get_word_vector(i).tolist() for i in entities}
	print (len(vectors))
	with open(outfile_name, "w") as outfile:
		json.dump(vectors, outfile)

if __name__ == "__main__":

	# we just extract all the possible words we could think of
	# we don't need all of them later, but it cannot hurt.

	word_list = set()
	experiments = ['our_survey', 'ggw', 'corporate_insecthood']

	preprocess = {'slaves': 'slave', 'africans': 'african', 'indians': 'indian', 'italians': 'italian', 'irishmen': 'irish', 'jews': 'jewish', 'catholics': 'catholic', 'muslims': 'muslim', 'arabs': 'arab', 'immigrants': 'immigrant', 'mexicans': 'mexican'}

	for exp in experiments:
		entities = load_data(experiment=exp)
		word_list.update(entities)
		word_list.update([preprocess[e] for e in entities if e in preprocess])


	# from earlier experiments
	entities = ['fetus','baby','you','coma','deceased', 'machine', 'table', 'cup', 'rock', 'mountain', 'door', 'computer', 'calculator',  'gold','man', 'woman', 'girl', 'boy', 'she', 'he', 'his',  'her', 'dog', 'lion', 'frog',  'turtle', 'bug', 'spider', 'owl', 'mouse', 'infant',  'robot', 'god', 'ant'] 
	word_list.update(entities)


	agency = ["change", "cause", "aim", "intentional", "harm", "help", "decide", "control", "moral", "plan", "communicate", "think", "choose", "deliberate", "create", "guilty", "responsible", "act", "do", "express"]
	patiency = ["pain", "pleasure", "enraged", "afraid", "hunger", "thirsty", "sad", "proud", "embarrassed", "joy", "angry", "happy", "conscious", "aware", "experience", "imagine", "awake", "suffer", "enjoy", "desire"]

	word_list.update(agency)
	word_list.update(patiency)


	agency = ["control", "morality", "memory", "emotion", "recognition", "plan", "communicate", "think"]
	patiency = ["hungry", "afraid", "pain", "pleasant", "angry", "desire", "personality", "concious", "proud", "embarrassed", "joyful"]
	word_list.update(agency)
	word_list.update(patiency)

	agency = ["agency", "think", "judge", "reason", "decide", "act"]
	patiency = ["patiency", "aware", "emotion", "feel", "experience"]

	word_list.update(agency)
	word_list.update(patiency)


	animals = ["owl","turtle","frog", "spider","lion","squirrel","cat", "chimpanzee", "dog", "ape", "chimp", "primate", "gorilla"]
	machines = ["calculator","computer","machine", "robot", "android", "cyborg", "algorithm"]
	humans = ["you", "he", "his", "she", "her", "mother", "father", "boy", "girl", "woman", "man", "baby", "murderer", "patient"]
	debated_persons = ["deceased", "fetus", "god", "coma", "zombie","psychopath","disabled","alzheimer","dementia"] 
	word_list.update(animals + machines + humans + debated_persons)
	word_list.add("comatose")
	word_list.add("angle")
	word_list.add("angel")
	word_list.add("whistle")
	word_list.add("plough")
	word_list.add("irishman")
	word_list.add("plow")
	word_list.add("irish")

	modes = ["glove-coha", "word2vec-coha", "fasttext-coha", "gigaword-300d"]
	path_directory = "word-vector-files"
	os.makedirs(path_directory, exist_ok=True)

	for mode in modes:
		extract_vectors(word_list, os.path.join(path_directory, mode + ".json"), mode=mode)



