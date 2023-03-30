from utils import *
from scipy.stats import spearmanr

def top_and_bottom_language():
	word_vectors = load_word_vectors("gigaword-300d")
	entities, _, _, _, _ = load_data("Survey All")
	entities = preprocess(entities)
	agency, patiency = get_word_lists()

	agency_vector = get_embeddings_vector(word_vectors, agency)
	patiency_vector = get_embeddings_vector(word_vectors, patiency)

	agency_results = compute_word_scores(word_vectors, entities, agency_vector)
	patiency_results = compute_word_scores(word_vectors, entities, patiency_vector)

	diff = [i - j for i,j in zip(agency_results, patiency_results)]
	sum_ = [i + j for i,j in zip(agency_results, patiency_results)]

	df = pd.DataFrame(list(zip(entities, agency_results, patiency_results, diff, sum_)), columns=["item", "agency", "patiency", "diff", "sum"])
	cols = ["agency", "patiency", "diff", "sum"]

	print (" & ".join(["Measure", "Top 10 Entities", "Bottom 10 Entities"]), r"\\", r"\midrule")
	for col in cols:
		df = df.sort_values(by=[col], ascending=False)
		print (col + " & " + " ".join(df.item.tolist()[:10]) + " & " + " ".join(df.item.tolist()[-10:]) + r"\\ " + r"\midrule")

def histograms():
	word_vectors = load_word_vectors("gigaword-300d")
	entities, _, _, _, _ = load_data("Survey All")
	entities = preprocess(entities)
	agency, patiency = get_word_lists()

	agency_vector = get_embeddings_vector(word_vectors, agency)
	patiency_vector = get_embeddings_vector(word_vectors, patiency)

	agency_results = compute_word_scores(word_vectors, entities, agency_vector)
	patiency_results = compute_word_scores(word_vectors, entities, patiency_vector)

	diff = [i-j for i,j in zip(agency_results, patiency_results)]
	sum_ = [i+j for i,j in zip(agency_results, patiency_results)]

	df = pd.DataFrame(list(zip(entities, agency_results, patiency_results, diff, sum_)), columns=["item", "agency", "patiency", "diff", "sum"])
	
	def make_histograms(df):
		import matplotlib.pyplot as plt
		import os
		print (df.columns)
		df.columns = ['item', 'agency', 'experience', 'diff', 'sum']
		cols = ["agency", "experience", "diff", "sum"]
		os.makedirs("histograms-language", exist_ok=True)
		for col in cols:
			df.hist(column=col)
			#plt.show()
			plt.savefig("histograms-language/hist_language_" + col + ".pdf", format="pdf")
	make_histograms(df)

def language_correlations(experiment="Survey Top 31"):
	entities, _, _, _, _ = load_data(experiment)
	entities = preprocess(entities)
	agency, patiency = get_word_lists()

	all_results = []
	for embeddings_type in ["glove-coha", "word2vec-coha", "fasttext-coha"]:
		word_vectors = load_word_vectors(embeddings_type)
		agency_vector = get_embeddings_vector(word_vectors, agency)
		patiency_vector = get_embeddings_vector(word_vectors, patiency)
		agency_results = compute_word_scores(word_vectors, entities, agency_vector)
		patiency_results = compute_word_scores(word_vectors, entities, patiency_vector)
		all_results.append(agency_results)
		all_results.append(patiency_results)

	out = []
	columns = ["Agency GloVe", "Agency Word2Vec", "Agency FastText", "Patiency GloVe", "Patiency Word2Vec", "Patiency FastText"]
	print (r"& & Agency GloVe & Agency Word2Vec & Agency FastText & Patiency GloVe & Patiency Word2Vec & Patiency FastText \\ \midrule ")
	if experiment == "Survey Top 31":
		print (r"\multirow{6}{*}{Top 31 Entities}")
	else:
		print (r"\multirow{6}{*}{Whole Survey}")
	c_1, c_2 = 0, 0
	for i, col in zip(all_results, columns):
		row = []
		c_2 = 0
		for j in all_results:
			res = spearmanr(i,j)
			corr, pvalue = res.correlation, res.pvalue
			corr = str(np.round(corr, 2))
			if c_1 > c_2:
				row.append("-")
			else:
				row.append(corr)
			c_2 += 1
		c_1 += 1
		print ( " & " + col + " & " + " & ".join(row) + " " + r"\\")


def top_and_bottom_entities_language():
	word_vectors = load_word_vectors("gigaword-300d")
	entities, _, _, _, _ = load_data("Survey Top 31")
	entities = preprocess(entities)
	agency, patiency = get_word_lists()

	agency_vector = get_embeddings_vector(word_vectors, agency)
	patiency_vector = get_embeddings_vector(word_vectors, patiency)

	agency_results = compute_word_scores(word_vectors, entities, agency_vector)
	patiency_results = compute_word_scores(word_vectors, entities, patiency_vector)

	df = pd.DataFrame(list(zip(entities, agency_results, patiency_results)), columns=["item", "agency", "patiency"])
	cols = ["agency", "patiency"]

	df["agency_rank"] = df.agency.rank(method='first')
	df["patiency_rank"] = df.patiency.rank(method='first')
	df["diff"] = df.agency - df.patiency
	"""
	df = df.sort_values(by=["agency"])
	for i, row in df.iterrows():
		print (row)
	sys.exit(0)
	"""
	df["diff_ranks"] = df.agency_rank - df.patiency_rank
	#df["diff_ranks"] = df.diff_ranks.apply(abs)
	#df = df.sort_values(by=["diff_ranks"])
	df = df.sort_values(by=["diff_ranks"])
	for i,j,k in zip(df.item, df["diff_ranks"], df["diff"]):
		print (i, " & ", int(j), " & ", np.round(k, 2), r"\\")


if __name__ == "__main__":
	#top_and_bottom_language()
	histograms()
	#language_correlations()
	#language_correlations(experiment="Survey All")
	#top_and_bottom_entities_language()
