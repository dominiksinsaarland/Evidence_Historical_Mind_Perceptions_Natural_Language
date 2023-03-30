from utils import *
from scipy.stats import spearmanr

def main_evaluation_metrics():
	out_str = "N & Embeddings & Survey Ratings & Agency & Patiency " + r"\\" + "\n"
	agency, patiency = get_word_lists()
	for experiment in ["Survey Top 31", "Survey Top 58", "Survey All"]:
		for embeddings_type in ["glove-coha", "word2vec-coha", "fasttext-coha"]:
			for adjusted in [True, False]:
				entities, agency_survey, patiency_survey, diff_survey, sum_survey = load_data(experiment=experiment, adjusted=adjusted)
				entities = preprocess(entities)
				word_vectors = load_word_vectors(embeddings_type)
				agency_vector = get_embeddings_vector(word_vectors, agency)
				patiency_vector = get_embeddings_vector(word_vectors, patiency)
				agency_results = compute_word_scores(word_vectors, entities, agency_vector)
				patiency_results = compute_word_scores(word_vectors, entities, patiency_vector)
				n = str(len(entities))
				if adjusted:
					out_str += n + " & " + embeddings_type + " & adjusted "
				else:
					out_str += n + " & " + embeddings_type + " & unadjusted "
				res = spearmanr(agency_survey, agency_results)
				corr, pvalue = res.correlation, res.pvalue
				corr = str(np.round(corr, 2))
				pvalue = str(np.round(pvalue, 3))
				corr = corr.replace("0.", ".")
				#print ("patiency", year, "correlation", corr, "(p value: " + pvalue + ")")
				out_str += " & " + corr + " (" + pvalue + ")"
				out_str += r"\\" + "\n"
		out_str += r" \midrule" + "\n"
	print (out_str)

def robustness_to_different_pretraining_corpora():
	out_str = "Algorithm & Training Corpus & Agency & Patiency " + r"\\" + "\n"
	agency, patiency = get_word_lists()
	corpora = ["Wiki + GigaWord (6B tokens)", "CommonCrawl (840B tokens)", "COHA (80M tokens)", "GoogleNews Corpus (100B tokens)", "COHA (80M tokens)", "CommonCrawl (840B tokens)", "COHA (80M tokens)"]

	embeddings = ["gigaword-300d", "glove-commoncrawl", "glove-coha", "google-newsvectors", "word2vec-coha", "fasttext-commoncrawl", "fasttext-coha"]
	for embeddings_type, corpus in zip(embeddings, corpora):
		entities, agency_survey, patiency_survey, diff_survey, sum_survey = load_data(experiment="Survey All", adjusted=True)
		entities = preprocess(entities)
		if embeddings_type == "google-newsvectors":
			entities = preprocess_word2vec(entities)
		word_vectors = load_word_vectors(embeddings_type)
		agency_vector = get_embeddings_vector(word_vectors, agency)
		patiency_vector = get_embeddings_vector(word_vectors, patiency)
		agency_results = compute_word_scores(word_vectors, entities, agency_vector)
		patiency_results = compute_word_scores(word_vectors, entities, patiency_vector)

		out_str += embeddings_type.split("-")[0] + " & " + corpus

		res = spearmanr(agency_survey, agency_results)
		corr, pvalue = res.correlation, res.pvalue
		corr = str(np.round(corr, 2))
		pvalue = str(np.round(pvalue, 3))
		corr = corr.replace("0.", ".")
		out_str += " & " + corr + " (" + pvalue + ")"
		#print ("agency", year, "correlation", corr, "(p value: " + pvalue + ")")

		res = spearmanr(patiency_survey, patiency_results)
		corr, pvalue = res.correlation, res.pvalue
		corr = str(np.round(corr, 2))
		pvalue = str(np.round(pvalue, 3))
		corr = corr.replace("0.", ".")
		#print ("patiency", year, "correlation", corr, "(p value: " + pvalue + ")")
		out_str += " & " + corr + " (" + pvalue + ")"
		out_str += r"\\" + "\n"
	print (out_str)


def robustness_to_window_and_embeddings_size():
	# have to run on cluster, because embeddings are stored there

	out_str = "Window Size & Vector Size & Agency & Patiency " + r"\\" + "\n"
	agency, patiency = get_word_lists()

	entities, agency_survey, patiency_survey, diff_survey, sum_survey = load_data(experiment="Survey All", adjusted=True)
	entities = preprocess(entities)

	window_sizes = [5,10,15]
	vector_dims = [50, 100, 200]
	path_coha = "../longitudinal_study/final_models_coha_2000"
	for window_size in window_sizes:
		for vector_dim in vector_dims:
			words = agency + patiency + entities
			path = "/cluster/work/lawecon/Work/dominik/glove/glove_20xx_" + str(vector_dim) + "_" + str(window_size) + "/vectors.txt"
			word_vectors = load_glove_vectors(path, words)

			agency_vector = get_embeddings_vector(word_vectors, agency)
			patiency_vector = get_embeddings_vector(word_vectors, patiency)
			agency_results = compute_word_scores(word_vectors, entities, agency_vector)
			patiency_results = compute_word_scores(word_vectors, entities, patiency_vector)
			out_str += str(window_size) + " & " + str(vector_dim)

			res = spearmanr(agency_survey, agency_results)
			corr, pvalue = res.correlation, res.pvalue
			corr = str(np.round(corr, 2))
			pvalue = str(np.round(pvalue, 3))
			corr = corr.replace("0.", ".")
			out_str += " & " + corr + " (" + pvalue + ")"
			res = spearmanr(patiency_survey, patiency_results)
			corr, pvalue = res.correlation, res.pvalue
			corr = str(np.round(corr, 2))
			pvalue = str(np.round(pvalue, 3))
			corr = corr.replace("0.", ".")
			out_str += " & " + corr + " (" + pvalue + ")"
			out_str += r"\\" + "\n"

	print (out_str)


def sum_and_diff():
	out_str = "Embeddings & Difference &  Summed " + r"\\" + "\n"
	agency, patiency = get_word_lists()
	entities, agency_survey, patiency_survey, diff_survey, sum_survey = load_data(experiment="Survey All", adjusted=False)
	entities = preprocess(entities)
	for embeddings_type in ["glove-coha", "word2vec-coha", "fasttext-coha"]:
		word_vectors = load_word_vectors(embeddings_type)
		agency_vector = get_embeddings_vector(word_vectors, agency)
		patiency_vector = get_embeddings_vector(word_vectors, patiency)
		agency_results = compute_word_scores(word_vectors, entities, agency_vector)
		patiency_results = compute_word_scores(word_vectors, entities, patiency_vector)
		diff = [i-j for i,j in zip(agency_results, patiency_results)]
		sum_ = [i+j for i,j in zip(agency_results, patiency_results)]
		out_str += embeddings_type
		res = spearmanr(diff_survey, diff)
		corr, pvalue = res.correlation, res.pvalue
		corr = str(np.round(corr, 2))
		pvalue = str(np.round(pvalue, 3))
		corr = corr.replace("0.", ".")
		out_str += " & " + corr + " (" + pvalue + ")"
		res = spearmanr(sum_survey, sum_)
		corr, pvalue = res.correlation, res.pvalue
		corr = str(np.round(corr, 2))
		pvalue = str(np.round(pvalue, 3))
		corr = corr.replace("0.", ".")
		out_str += " & " + corr + " (" + pvalue + ")" + "\n"
	print (out_str)

def other_surveys():
	out_str = "Entity Rating Survey & Embeddings & Agency/Patiency Words & Agency & Patiency" + r"\\" + "\n"
	experiments = ["GGW", "Corp. Insecthood"]
	for experiment in experiments:
		for embeddings_type in ["glove-coha", "word2vec-coha", "fasttext-coha"]:
			for word_list in ["GGW", "Corp. Insecthood"]:
				if word_list != experiment:
					continue

				entities, agency_survey, patiency_survey, diff_survey, sum_survey = load_data(experiment=experiment, adjusted=False)
				agency, patiency = get_word_lists(mode=word_list)
				word_vectors = load_word_vectors(embeddings_type)
				agency_vector = get_embeddings_vector(word_vectors, agency)
				patiency_vector = get_embeddings_vector(word_vectors, patiency)
				agency_results = compute_word_scores(word_vectors, entities, agency_vector)
				patiency_results = compute_word_scores(word_vectors, entities, patiency_vector)

				out_str += experiment + " Survey & " + embeddings_type + " & " + word_list
				res = spearmanr(agency_survey, agency_results)
				corr, pvalue = res.correlation, res.pvalue
				corr = str(np.round(corr, 2))
				pvalue = str(np.round(pvalue, 3))
				corr = corr.replace("0.", ".")
				out_str += " & " + corr + " (" + pvalue + ")"
				res = spearmanr(patiency_survey, patiency_results)
				corr, pvalue = res.correlation, res.pvalue
				corr = str(np.round(corr, 2))
				pvalue = str(np.round(pvalue, 3))
				corr = corr.replace("0.", ".")
				out_str += " & " + corr + " (" + pvalue + ")"
				out_str += r"\\" + "\n"
	print (out_str)

#ok, we're basically done...
def diff_language_survey():
	agency, patiency = get_word_lists()
	entities, agency_survey, patiency_survey, diff_survey, sum_survey = load_data(experiment="Survey Top 31", adjusted=True)

	df_survey = pd.read_csv("data/items-agg.csv")
	word_vectors = load_word_vectors("gigaword-300d")
	agency_vector = get_embeddings_vector(word_vectors, agency)
	patiency_vector = get_embeddings_vector(word_vectors, patiency)
	agency_results = compute_word_scores(word_vectors, entities, agency_vector)
	patiency_results = compute_word_scores(word_vectors, entities, patiency_vector)

	df = pd.DataFrame(list(zip(entities, agency_results, patiency_results)), columns=["item", "agency_language", "patiency_language"])
	df["agency_language_rank"] = df.agency_language.rank(method='first')
	df["patiency_language_rank"] = df.patiency_language.rank(method='first')

	df = pd.merge(df, df_survey,on=["item"])

	df["agency_rank"] = df.agency.rank(method='first')
	df["patiency_rank"] = df.patiency.rank(method='first')

	df["diff_ranks"] = df.agency_language_rank - df.agency_rank
	df = df.sort_values(by=["diff_ranks"])
	for i,j in zip(df.item, df["diff_ranks"]):#, df["diff"]):
		print (i, " & ", int(j), r"\\")

	print ("*" * 100)

	df["diff_ranks"] = df.patiency_language_rank - df.patiency_rank
	df = df.sort_values(by=["diff_ranks"])
	for i,j in zip(df.item, df["diff_ranks"]):#, df["diff"]):
		print (i, " & ", int(j), r"\\")



if __name__ == "__main__":
	#main_evaluation_metrics()
	#robustness_to_different_pretraining_corpora()
	#robustness_to_window_and_embeddings_size()
	#sum_and_diff()
	#other_surveys()	
	diff_language_survey_agency()
