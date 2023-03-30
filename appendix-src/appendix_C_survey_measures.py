from scipy.stats import spearmanr
import pandas as pd
import numpy as np


def correlation_table():
	df = pd.read_csv("data/items-agg.csv")
	cols = ["agency", "agency_adj", "patiency", "patiency_adj"]
	print (r"& & agency & agency\_adj & patiency & patiency\_adj \\ \midrule ")
	print (r"\multirow{4}{*}{Whole Survey}")
	for col in cols:
		row = []
		for col_2 in cols:
			corr = spearmanr(df[col], df[col_2]).correlation
			corr = str(np.round(corr, 2))
			row.append(corr) 
		print ( " & " + col + " & " + " & ".join(row) + " " + r"\\ \rmidrule")

	entities_to_consider = ["human", "man", "woman", "boy", "girl", "father", "mother", "dad", "mom", "grandfather", "grandmother", "baby", "infant", "fetus", "corpse", "dog", "puppy", "cat", "kitten", "frog", "ant", "fish", "mouse", "bird", "shark", "elephant", "beetle", "insect", "chimpanzee", "monkey", "primate"]
	df = df[df["item"].isin(entities_to_consider)]
	print (r"& & agency & agency\_adj & patiency & patiency\_adj \\ \midrule ")
	print (r"\multirow{4}{*}{Top 31 Entities}")
	for col in cols:
		row = []
		for col_2 in cols:
			corr = spearmanr(df[col], df[col_2]).correlation
			corr = str(np.round(corr, 2))
			row.append(corr) 		
		print ( " & " + col + " & " + " & ".join(row) + " " + r"\\")
	print (r"\bottomrule")



	cols = ["agency_adj", "patiency_adj", "diff_adj", "sum_adj"]
	print ( " & " + " & ".join(cols) + r"\\")
	for col in cols:
		row = []
		for col_2 in cols:
			corr = spearmanr(df[col], df[col_2]).correlation
			corr = str(np.round(corr, 2))
			row.append(corr) 	
		print (col + " & " + " & ".join(row) + " " + r"\\")

def make_histograms():
	import matplotlib.pyplot as plt
	import os
	df = pd.read_csv("data/items-agg.csv")
	cols = ["agency", "experience", "diff", "sum"]
	df.rename(columns = {'patiency':'experience'}, inplace = True)
	print (df.columns)
	os.makedirs("histograms", exist_ok=True)
	for col in cols:
		df.hist(column=col)
		#plt.show()
		plt.savefig("histograms/hist_" + col + ".pdf", format="pdf")

def top_and_bottom_survey():
	df = pd.read_csv("data/items-agg.csv")
	cols = ["agency", "patiency", "diff", "sum"]
	print (" & ".join(["Measure", "Top 10 Entities", "Bottom 10 Entities"]), r"\\", r"\midrule")
	for col in cols:
		df = df.sort_values(by=[col], ascending=False)
		print (col + " & " + " ".join(df.item.tolist()[:10]) + " & " + " ".join(df.item.tolist()[-10:]) + r"\\ " + r"\midrule")


def sort_by_difference():
	df = pd.read_csv("data/items-agg.csv")
	entities_to_consider = ["human", "man", "woman", "boy", "girl", "father", "mother", "dad", "mom", "grandfather", "grandmother", "baby", "infant", "fetus", "corpse", "dog", "puppy", "cat", "kitten", "frog", "ant", "fish", "mouse", "bird", "shark", "elephant", "beetle", "insect", "chimpanzee", "monkey", "primate"]
	df = df[df["item"].isin(entities_to_consider)]
	df["agency_rank"] = df.agency.rank(method='first')
	df["patiency_rank"] = df.patiency.rank(method='first')
	df["diff_ranks"] = df.agency_rank - df.patiency_rank
	#df["diff_ranks"] = df.diff_ranks.apply(abs)
	df = df.sort_values(by=["diff_ranks"])
	for i,j,k in zip(df.item, df.diff_ranks, df["diff"]):
		print (i, " & ", int(j), " & ", np.round(k, 2), r"\\")

if __name__ == "__main__":
	# makes the figures
	#correlation_table()
	make_histograms()
	#sort_by_difference()
	#top_and_bottom_survey()

