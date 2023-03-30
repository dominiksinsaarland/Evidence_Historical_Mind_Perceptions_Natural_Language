import numpy as np
import json
import pandas as pd

def load_data(experiment, adjusted=True):
	# load survey data, i.e. survey entities and scores
	if experiment == "Survey All":
		df = pd.read_csv("../appendix_paper/data/items-agg.csv")
		entities = df["item"].tolist()
		if adjusted:
			agency, patiency, diff_, sum_ = df["agency_adj"].tolist(), df["patiency_adj"].tolist(), df["diff_adj"].tolist(), df["sum_adj"].tolist()
		else:
			agency, patiency, diff_, sum_ = df["agency"].tolist(), df["patiency"].tolist(), df["diff"].tolist(), df["sum"].tolist()
		return entities, agency, patiency, diff_, sum_
	elif experiment == "Survey Top 31":
		# pre-registered 31 entities
		df = pd.read_csv("../appendix_paper/data/items-agg.csv")
		entities_to_consider = ["human", "man", "woman", "boy", "girl", "father", "mother", "dad", "mom", "grandfather", "grandmother", "baby", "infant", "fetus", "corpse", "dog", "puppy", "cat", "kitten", "frog", "ant", "fish", "mouse", "bird", "shark", "elephant", "beetle", "insect", "chimpanzee", "monkey", "primate"]
		df = df[df["item"].isin(entities_to_consider)]
		entities = df["item"].tolist()
		if adjusted:
			agency, patiency, diff_, sum_ = df["agency_adj"].tolist(), df["patiency_adj"].tolist(), df["diff_adj"].tolist(), df["sum_adj"].tolist()
		else:
			agency, patiency, diff_, sum_ = df["agency"].tolist(), df["patiency"].tolist(), df["diff"].tolist(), df["sum"].tolist()
		return entities, agency, patiency, diff_, sum_
	elif experiment == "Survey Top 58":
		# pre-registered 58 entities
		df = pd.read_csv("../appendix_paper/data/items-agg.csv")
		entities_to_consider = ["human", "man", "woman", "boy", "girl", "father", "mother", "dad", "mom", "grandfather", "grandmother", "baby", "infant", "fetus", "corpse", "dog", "puppy", "cat", "kitten", "frog", "ant", "fish", "mouse", "bird", "shark", "elephant", "beetle", "insect", "chimpanzee", "monkey", "primate", "car", "rock", "hammer", "computer", "robot", "god", "angle", "ghost", "puppet", "pigeon", "chicken", "rabbit", "fox", "turkey", "pig", "cow", "horse", "sheep", "lamb", "cucumber", "lettuce", "potato", "cabbage", "chocolate", "coffee", "tea", "butter"]
		df = df[df["item"].isin(entities_to_consider)]
		entities = df["item"].tolist()
		if adjusted:
			agency, patiency, diff_, sum_ = df["agency_adj"].tolist(), df["patiency_adj"].tolist(), df["diff_adj"].tolist(), df["sum_adj"].tolist()
		else:
			agency, patiency, diff_, sum_ = df["agency"].tolist(), df["patiency"].tolist(), df["diff"].tolist(), df["sum"].tolist()
		return entities, agency, patiency, diff_, sum_
	elif experiment == "GGW":
		df = pd.read_csv("data/gray_wegner_data.csv")
		entities, agency, patiency = df["Entity"].tolist(), df["GGW_Agency"].tolist(), df["GGW_Experience"].tolist()
		diff_ = [i - j for i,j in zip(agency, patiency)]
		sum_ = [i + j for i,j in zip(agency, patiency)]
		# replace multiword expressions
		entities[9] = "comatose"
		entities[2] = "deceased"
		return entities, agency, patiency, diff_, sum_
	elif experiment == "Corp. Insecthood":
		# Corporate Insecthood data
		df = pd.read_csv("data/corporate_insecthood_data.csv")
		# only non-corporate entities
		df = df[(df["Category"] == "Artifact / object") | (df["Category"] == "Non-human life") | (df["Category"] == "Human")]
		#personhood = df["Personhood_mean"].tolist()
		agency = df["Agency_mean"].tolist()
		patiency = df["Patiency_mean"].tolist()
		entities = [e.lower() for e in df["Entity"]]
		agency = [float(i.split()[0]) for i in agency]
		patiency = [float(i.split()[0]) for i in patiency]
		diff_ = [i - j for i,j in zip(agency, patiency)]
		sum_ = [i + j for i,j in zip(agency, patiency)]
		# replace multiword expressions
		index = entities.index("deceased man")
		entities[index] = "deceased"
		index = entities.index("patient in a persistent vegetative state")
		entities[index] = "comatose"
		return entities, agency, patiency, diff_, sum_

def projection_score(a,b, mode="cosine similarity"):
	# return cosine similarity between two word vectors
	if mode == "cosine similarity":
		return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
	elif mode == "euclidian distance":
		return np.linalg.norm(np.array(b)-np.array(a))
	elif mode == "dot product":
		return np.dot(a,b)

def compute_word_scores(word_vectors, entities, vector, mode="normal", anti_vector=None):
	# compute the similarity scores for all words and a given concept vector
	return [projection_score(word_vectors[e],vector) if e in word_vectors else np.nan for e in entities]


def get_embeddings_vector(word_vectors, words):
	# get the averaged embedding vectors given a set of words
	embeddings = np.array([word_vectors[w] for w in words if w in word_vectors])
	embeddings = np.nanmean(embeddings, axis=0)
	return embeddings

def get_word_lists(mode="Survey Wordlist"):
	# TODO implement
	if mode == "Survey Wordlist":
		agency = ["change", "cause", "aim", "intentional", "harm", "help", "decide", "control", "moral", "plan", "communicate", "think", "choose", "deliberate", "create", "guilty", "responsible", "act", "do", "express"]
		patiency = ["pain", "pleasure", "enraged", "afraid", "hunger", "thirsty", "sad", "proud", "embarrassed", "joy", "angry", "happy", "conscious", "aware", "experience", "imagine", "awake", "suffer", "enjoy", "desire"]
	elif mode == "GGW":
		agency = ["control", "morality", "memory", "emotion", "recognition", "plan", "communicate", "think"]
		patiency = ["hungry", "afraid", "pain", "pleasant", "angry", "desire", "personality", "concious", "proud", "embarrassed", "joyful"]
	elif mode == "Corp. Insecthood":
		"""
		from Corporate Insecthood Paper: We included items related to agency ( MORAL AGENCY ; THINKING ; DECISION - MAKING ) SELF - AWARENESS ), and patiency ( MORAL PATIENCY ; FEELING ; as previous research indicates are the major decomposable dimensions of morality and mind.
		
		moral agency: To what extent is ENTITY X capable of wronging others?
		thinking: To what extent is ENTITY X capable of thinking, judgment, and reason?
		decision-making: To what extent is ENTITY X capable of making decisions and acting on them?

		moral patiency: To what extent is ENTITY X capable of being wronged by others?
		feeling: To what extent is ENTITY X capable of emotions, feelings, and experiences?
		from these items, I selected the most salient words, and constructed the two wordlists:
		"""
		agency = ["agency", "think", "judge", "reason", "decide", "act"]
		patiency = ["patiency", "aware", "emotion", "feel", "experience"]


	return agency, patiency

# redo
def load_word_vectors(mode="gigaword-300d"):
	#modes = ["glove-commoncrawl", "gigaword-300d", "google-newsvectors", "fasttext-1mio-news", "fasttext-commoncrawl", "fasttext-commoncrawl-subwords", "glove-coha", "word2vec-coha", "fasttext-coha"]
	with open("word-vector-files/" + mode + ".json" ) as f:
		word_vectors = json.load(f)
	return word_vectors

def preprocess(entities):
	to_preprocess = {'slaves': 'slave', 'africans': 'african', 'indians': 'indian', 'italians': 'italian', 'irishmen': 'irish', 'jews': 'jewish', 'catholics': 'catholic', 'muslims': 'muslim', 'arabs': 'arab', 'immigrants': 'immigrant', 'mexicans': 'mexican', "angle": "angel"}
	for i, e in enumerate(entities): 
		if e in to_preprocess:
			entities[i] = to_preprocess[e]
	return entities

def preprocess_word2vec(entities):
	if "plough" in entities:
		index = entities.index("plough")
		entities[index] = "plow"
	if "irishmen" in entities:
		index = entities.index("irishmen")
		entities[index] = "irish"
	return entities

def load_glove_vectors(path, words):
	set_words = set(words)
	word_vectors = {}
	with open(path) as f:
		for line in f:
			line = line.strip().split()
			if line[0] in set_words:
				word_vectors[line[0]] = list(map(float, line[1:]))
	return word_vectors


