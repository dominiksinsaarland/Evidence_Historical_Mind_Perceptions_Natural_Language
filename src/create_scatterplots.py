from scipy.stats import spearmanr
import pandas as pd
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import fasttext
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from gensim.models import KeyedVectors
from utils import *
import os

def scatterplots(word_vectors="gigaword-300d"):
	agency, patiency = get_word_lists()
	entities, agency_survey, patiency_survey, diff_survey, sum_survey = load_data(experiment="Survey Top 31", adjusted=False)
	if word_vectors == "gigaword-300d":
		word_vectors = load_word_vectors("gigaword-300d")
	elif word_vectors == "word2vec-coha":
		word_vectors = load_word_vectors("gigaword-300d")

	agency_vector = get_embeddings_vector(word_vectors, agency)
	patiency_vector = get_embeddings_vector(word_vectors, patiency)
	agency_results = compute_word_scores(word_vectors, entities, agency_vector)
	patiency_results = compute_word_scores(word_vectors, entities, patiency_vector)


	os.makedirs("scatterplots", exist_ok=True)
	p_val, r_squared = plot(entities, agency_results, agency_survey, "agency", "identity", "ours_top_31")
	p_val, r_squared = plot(entities, patiency_results, patiency_survey, "experience", "identity", "ours_top_31")
          

def plot(entities, word_scores, survey_scores, dimension, mode, experiment):
	# wrapper function to plot stuff

	word_scores, survey_scores = survey_scores, word_scores

	if mode == "rank":
		word_ranks = np.zeros(len(word_scores))
		survey_ranks = np.zeros(len(survey_scores))
		for c, i in enumerate(np.argsort(word_scores)):
			word_ranks[i] = c+1
		word_scores = word_ranks
		for c, i in enumerate(np.argsort(survey_scores)):
			survey_ranks[i] = c+1
		survey_scores = survey_ranks
	elif mode == "log":
		word_scores = [np.log(i) for i in word_scores]
		survey_scores = [np.log(i) for i in survey_scores]

	import statsmodels.api as sm
	X = survey_scores
	y = word_scores

	X = sm.add_constant(X)
	results = sm.OLS(y, X).fit()

	intercept, coef = results.params

	res = spearmanr(word_scores, survey_scores)
	corr, pvalue = res.correlation, res.pvalue
	corr = str(np.round(corr, 2))

	lst = list(zip(entities, word_scores, survey_scores))
	df = pd.DataFrame(lst, columns =['entity', 'computed_result', 'survey_result'])
	#print (df)

	plt.tight_layout()
	plt.style.use('seaborn-whitegrid')
	fig, ax = plt.subplots(1)

	if "31" in experiment:
		plt.scatter(df["survey_result"], df["computed_result"])
		print (min(survey_scores), max(survey_scores), abs(min(survey_scores)) + abs(max(survey_scores)))
		x_scale = (abs(min(survey_scores)) + abs(max(survey_scores))) / 1000
		reg_x = np.arange(min(survey_scores), max(survey_scores), x_scale)
		print (coef, intercept)
		reg_y = reg_x * coef + intercept
		#plt.plot(reg_x, reg_y, label="r^2=" + str(np.round(results.rsquared, 2)) + "\nspearman's rho=" + corr)
		plt.plot(reg_x, reg_y, label="spearman's rho=" + corr)
		for c, (txt, i,j) in enumerate(zip(entities, df["survey_result"], df["computed_result"])):
			ax.annotate(txt, (i,j),  xytext= (i, j), rotation=45, fontsize=6, va="top")



	else:
		sns.regplot(df["survey_result"], df["computed_result"])

	plt.xlabel(dimension.title() + " Score from Language Model")
	plt.ylabel(dimension.title() + "  Score from Human Survey")
	plt.legend(loc="upper left")
	plt.savefig("scatterplots/new_word2vec_coha_scatter_plot_" + experiment + "_" + dimension + "_" + mode + ".pdf", format="pdf")
	plt.clf()
	plt.close("all")
	return results.pvalues[1], results.rsquared

if __name__ == "__main__":
	scatterplots()
