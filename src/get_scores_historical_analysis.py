# right
import numpy as np
import argparse
#import fasttext
from tqdm import tqdm
import os
import pandas as pd
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

def load_entities(category="male"):
	if category == "male":
		# dad not in aligned models
		return ["boy", "grandfather", "man", "father", "dad"][:-1]
	elif category == "female":
		# mom not in algined models
		return ["girl", "grandmother", "woman", "mother", "mom"][:-1]
	elif category == "animal":
		return ["cat", "dog", "cow", "pig", "kitten", "lamb", "sheep", "horse", "goat"]
	elif category == "wild_animals":
		return ["rabbit", "worm", "mouse", "pigeon", "beetle", "fox", "chimpanzee", "frog", "chicken", "turkey", "bird", "fish", "elephant", "monkey", "ant", "snake", "primate", "insect", "shark", "bee"] 
	elif category == "control_humans":
		return ["boy", "grandfather", "man", "father", "dad"] + ["girl", "grandmother", "woman", "mother", "mom"]
	elif category == "control":
		entities = pd.read_csv("../data/items-agg.csv")["item"].tolist()
		return entities

def cosine_similarity(a,b):
	return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_embeddings_vector(word_vectors, words):
	embeddings = np.array([word_vectors[w] for w in words if w in word_vectors])
	embeddings = np.nanmean(embeddings, axis=0)
	return embeddings

def load_embeddings(year, words, folder):
	path = folder + "/" + str(year) + ".model"
	model = Word2Vec.load(path)
	word_vectors = {i:model.wv[i] for i in words if i in model.wv}
	return word_vectors


if __name__ == "__main__":
	# word lists
	agency = ["change", "cause", "aim", "intentional", "harm", "help", "decide", "control", "moral", "plan", "communicate", "think", "choose", "deliberate", "create", "guilty", "responsible", "act", "do", "express"]
	patiency = ["pain", "pleasure", "afraid", "hunger", "thirsty", "sad", "proud", "embarrassed", "joy", "angry", "happy", "conscious", "aware", "experience", "imagine", "awake", "suffer", "enjoy", "desire"]

	years = [str(i) for i in range(1820, 2020, 10)]
	male, female = load_entities("male"), load_entities("female")

	animals, control = load_entities("animal"), load_entities("wild_animals")
	# For creating our appendix robustness check analysis, run with the following line instead
	# animals, control = load_entities("animal"), load_entities("control")

	words = male + female + agency + patiency + animals + control
	folders = ["longitudinal_study/aligned_models"] + ["longitudinal_study/aligned_models_" + str(i) for i in range(1, 10)]

	out_gender_words = []
	out_animal_words = []
	# iterate over 10 word embedding models
	for folder in folders:
		y = []
		y_animal = []
		# for each model, get scores for all decades
		for year in tqdm(years):
			word_vectors = load_embeddings(year, words, folder)

			agency_vector = get_embeddings_vector(word_vectors, agency)
			patiency_vector = get_embeddings_vector(word_vectors, patiency)

			agency_male = np.nanmean([cosine_similarity(word_vectors[i], agency_vector) for i in male])
			patiency_male = np.nanmean([cosine_similarity(word_vectors[i], patiency_vector) for i in male])
			agency_female = np.nanmean([cosine_similarity(word_vectors[i], agency_vector) for i in female])
			patiency_female = np.nanmean([cosine_similarity(word_vectors[i], patiency_vector) for i in female])
			y.append((agency_male, patiency_male, agency_female, patiency_female))

			agency_animal = np.nanmean([cosine_similarity(word_vectors[i], agency_vector) for i in animals if i in word_vectors])
			patiency_animal = np.nanmean([cosine_similarity(word_vectors[i], patiency_vector) for i in animals  if i in word_vectors])
			agency_control = np.nanmean([cosine_similarity(word_vectors[i], agency_vector) for i in control if i in word_vectors])
			patiency_control = np.nanmean([cosine_similarity(word_vectors[i], patiency_vector) for i in control if i in word_vectors])
			y_animal.append((agency_animal, patiency_animal, agency_control, patiency_control))

		out_gender_words.append(y)
		out_animal_words.append(y_animal)
	np.save("bootstrapped_agency_patiency_scores_gender_words", np.array(out_gender_words))
	np.save("bootstrapped_agency_patiency_scores_animal_words", np.array(out_animal_words))
