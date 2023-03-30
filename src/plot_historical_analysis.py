	
#from utils import *
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import numpy as np
import copy
import argparse
from tqdm import tqdm
import seaborn as sns
import pandas as pd
#plt.style.use('seaborn-whitegrid')

def moving_average(a, window_size):
	if window_size == 0:
		return a
	out = []
	for i in range(len(a)):
		start = max(0, i - window_size)
		out.append(np.mean(a[start:i + 1]))
	return out

def plot(window_size = 1, outfile_name=""):
	colors= ["tab:red", "tab:blue"]
	years = [str(i) for i in range(1820, 2020, 10)]
	# male, female
	y = np.load("bootstrapped_agency_patiency_scores_gender_words.npy")
	male_agency, male_experience, female_agency, female_experience = y[:,:,0], y[:,:,1], y[:,:,2], y[:,:,3]

	# aestethics
	large = 22; med = 16; small = 12
	params = {'axes.titlesize': large,
		  'legend.fontsize': med,
		  'figure.figsize': (16, 10),
		  'axes.labelsize': med,
		  'axes.titlesize': med,
		  'xtick.labelsize': med,
		  'ytick.labelsize': med,
		  'figure.titlesize': large}
	plt.rcParams.update(params)
	plt.style.use('seaborn-whitegrid')
	sns.set_style("white")

	fig = plt.figure()
	ax = fig.add_subplot(111)

	agency_means, agency_lower, agency_upper = [], [], []
	experience_means, experience_lower, experience_upper = [], [], []
	y_1, y_2 = [], []

	agency_rating = np.array(female_agency) - np.array(male_agency)
	experience_rating = np.array(female_experience) - np.array(male_experience)
		
	# https://stackoverflow.com/questions/50161140/how-to-plot-a-time-series-array-with-confidence-intervals-displayed-in-python
	
	# save stuff
	for year in range(len(years)):
	
		sample = agency_rating[:,year]
		SD = sample.std()
		SE = SD / np.sqrt(len(sample) - 1)
		mean = sample.mean()
		agency_means.append(mean)
		agency_lower.append(mean - 1.96 * SE)
		agency_upper.append(mean + 1.96 * SE)
		sample = experience_rating[:,year]
		SD = sample.std()
		SE = SD / np.sqrt(len(sample) - 1)
		mean = sample.mean()
		experience_means.append(mean)
		experience_lower.append(mean - 1.96 * SE)
		experience_upper.append(mean + 1.96 * SE)

	label = "Agency"
	agency_means = moving_average(agency_means, window_size)
	agency_lower = moving_average(agency_lower, window_size)
	agency_upper = moving_average(agency_upper, window_size)
	
	plt.plot(np.arange(len(years)),agency_means, label=label, c="tab:red", linestyle="-", lw=0.7)
	plt.fill_between(np.arange(len(years)), agency_lower, agency_upper, color='tab:red', alpha=.1) #std curves.

	label = "Experience"
	experience_means = moving_average(experience_means, window_size)
	experience_lower = moving_average(experience_lower, window_size)
	experience_upper = moving_average(experience_upper, window_size)
	
	
	plt.plot(np.arange(len(years)),experience_means, label=label, c="tab:blue", linestyle="-", lw=0.7)
	plt.fill_between(np.arange(len(years)), experience_lower, experience_upper, color='tab:blue', alpha=.1) #std curves.

	plt.xticks(range(len(years)), years, rotation=45)
	ax.set_ylabel("Differences")

	plt.legend()
	plt.title("Female-Male Differences in Semantic Agency and Semantic Experience")
	if len(years) > 12:
		every_nth = len(years) // 12
		for n, label in enumerate(ax.xaxis.get_ticklabels()):
			if n % every_nth != 0:
				label.set_visible(False)
	# line at 0
	plt.plot(np.arange(len(years)), [0] * len(years), linestyle="dotted", lw=1.3, c="black")
	# Lighten borders
	plt.gca().spines["top"].set_alpha(.3)
	plt.gca().spines["bottom"].set_alpha(.3)
	plt.gca().spines["right"].set_alpha(.3)
	plt.gca().spines["left"].set_alpha(.3)
	plt.grid(alpha=0.5)
	plt.savefig("main_results_difference_female_minus_male_window_size_" + str(window_size) + ".pdf", format="pdf", bbox_inches="tight")

def plot_animals(filename="", outfile_name="", window_size=1):
	if not filename:
		sys.exit(0)
	if not outfile_name:
		sys.exit(0)
	colors= ["tab:red", "tab:blue"]
	years = [str(i) for i in range(1820, 2020, 10)]
	# male, female
	y = np.load(filename)
	male_agency, male_experience, female_agency, female_experience = y[:,:,0], y[:,:,1], y[:,:,2], y[:,:,3]

	# aestethics
	large = 22; med = 16; small = 12
	params = {'axes.titlesize': large,
		  'legend.fontsize': med,
		  'figure.figsize': (16, 10),
		  'axes.labelsize': med,
		  'axes.titlesize': med,
		  'xtick.labelsize': med,
		  'ytick.labelsize': med,
		  'figure.titlesize': large}
	plt.rcParams.update(params)
	plt.style.use('seaborn-whitegrid')
	sns.set_style("white")

	fig = plt.figure()
	ax = fig.add_subplot(111)

	agency_means, agency_lower, agency_upper = [], [], []
	experience_means, experience_lower, experience_upper = [], [], []
	y_1, y_2 = [], []

	agency_rating = np.array(male_agency) - np.array(female_agency)
	experience_rating = np.array(male_experience) - np.array(female_experience)
	"""
	save_output = []
	print (male_agency.shape)
	for i,j in zip(female_agency, male_agency):
		save_output.append(i.tolist() + j.tolist())
		#print (save_output[-1])
		print (len(save_output[-1]))
	df = pd.DataFrame(save_output, columns=[["female_" + str(i) for i in range(1820, 2020, 10)] + ["male_" + str(i) for i in range(1820, 2020, 10)]])
	print (df)
	results = []
	for year in range(1820, 2020, 10):
		# ...
		# 	df = pd.DataFrame(save_output, columns=[["female_agency_" + str(i) for i in range(1820, 2020, 10)] + ["male_agency_" + str(i) for i in range(1820, 2020, 10)]])
		female_col = "female_" + str(year)
		male_col = "male_" + str(year)
		diff_col = df[male_col].to_numpy() - df[female_col].to_numpy()
		diff_col = diff_col.squeeze()
		results.append(diff_col)
	results = np.array(results).T
	print (results.shape)
	results = results.tolist()	
	print ("agency")
	df = pd.DataFrame(results, columns=["t_" + str(i) for i in range(1820, 2020, 10)])
	print (df)
	df.to_csv("domesticated_animals_wild_animals_agency_differences_10_folds.csv", index=False)


	#TADA
	save_output = []
	print (male_agency.shape)
	for i,j in zip(female_experience, male_experience):
		save_output.append(i.tolist() + j.tolist())
		#print (save_output[-1])
		print (len(save_output[-1]))
	df = pd.DataFrame(save_output, columns=[["female_" + str(i) for i in range(1820, 2020, 10)] + ["male_" + str(i) for i in range(1820, 2020, 10)]])
	results = []
	for year in range(1820, 2020, 10):
		# ...
		# 	df = pd.DataFrame(save_output, columns=[["female_agency_" + str(i) for i in range(1820, 2020, 10)] + ["male_agency_" + str(i) for i in range(1820, 2020, 10)]])
		female_col = "female_" + str(year)
		male_col = "male_" + str(year)
		diff_col = df[male_col].to_numpy() - df[female_col].to_numpy()
		diff_col = diff_col.squeeze()
		results.append(diff_col)
	results = np.array(results).T
	print (results.shape)
	results = results.tolist()	
	df = pd.DataFrame(results, columns=["t_" + str(i) for i in range(1820, 2020, 10)])
	df.to_csv("domesticated_animals_wild_animals_experience_differences_10_folds.csv", index=False)
	print ("experience")
	print (df)



	sys.exit(0)
	"""
		
	# https://stackoverflow.com/questions/50161140/how-to-plot-a-time-series-array-with-confidence-intervals-displayed-in-python
	
	for year in range(len(years)):
		sample = agency_rating[:,year]
		SD = sample.std()
		SE = SD / np.sqrt(len(sample) - 1)
		mean = sample.mean()
		agency_means.append(mean)
		agency_lower.append(mean - 1.96 * SE)
		agency_upper.append(mean + 1.96 * SE)

		sample = experience_rating[:,year]
		SD = sample.std()
		SE = SD / np.sqrt(len(sample) - 1)
		mean = sample.mean()
		experience_means.append(mean)
		experience_lower.append(mean - 1.96 * SE)
		experience_upper.append(mean + 1.96 * SE)


	label = "Agency"
	agency_means = moving_average(agency_means, window_size)
	agency_lower = moving_average(agency_lower, window_size)
	agency_upper = moving_average(agency_upper, window_size)
	
	
	plt.plot(np.arange(len(years)),agency_means, label=label, c="tab:red", linestyle="-", lw=0.7)
	plt.fill_between(np.arange(len(years)), agency_lower, agency_upper, color='tab:red', alpha=.1) #std curves.

	label = "Experience"
	experience_means = moving_average(experience_means, window_size)
	experience_lower = moving_average(experience_lower, window_size)
	experience_upper = moving_average(experience_upper, window_size)
	plt.plot(np.arange(len(years)),experience_means, label=label, c="tab:blue", linestyle="-", lw=0.7)
	plt.fill_between(np.arange(len(years)), experience_lower, experience_upper, color='tab:blue', alpha=.1) #std curves.

	plt.xticks(range(len(years)), years, rotation=45)
	ax.set_ylabel("Differences")

	plt.legend(loc="upper left")
	plt.title("Animal-Control Differences in Semantic Agency and Semantic Experience")
	if len(years) > 12:
		every_nth = len(years) // 12
		for n, label in enumerate(ax.xaxis.get_ticklabels()):
			if n % every_nth != 0:
				label.set_visible(False)
	# line at 0
	plt.plot(np.arange(len(years)), [0] * len(years), linestyle="dotted", lw=1.3, c="black")
	# Lighten borders
	plt.gca().spines["top"].set_alpha(.3)
	plt.gca().spines["bottom"].set_alpha(.3)
	plt.gca().spines["right"].set_alpha(.3)
	plt.gca().spines["left"].set_alpha(.3)
	plt.grid(alpha=0.5)
	#plt.savefig("animal_words_over_time_with_control_difference_not_bootstrapped_v2.pdf", bbox_inches="tight", format="pdf") # old, with 1.96 * SD
	#plt.savefig("animal_words_over_time_with_control_difference_not_bootstrapped_v3.pdf", bbox_inches="tight", format="pdf") # old, with 1.96 * SE
	#plt.savefig(outfile_name + "_moving_average_" + str(window_size) + ".png", bbox_inches="tight")
	plt.savefig(outfile_name, bbox_inches="tight", format="pdf")
	
	#plt.savefig(outfile_name, bbox_inches="tight")
	


if __name__ == "__main__":
	plot(window_size=1)
	# for unsmoothed appendix version of this plot, uncomment the following
	# plot(window_size=0)
	
	plot_animals(filename="bootstrapped_agency_patiency_scores_animal_words.npy", outfile_name="main_results_domesticated_animals_vs_wild_animals_window_size_1.pdf")
	# for unsmoothed appendix version of this plot, uncomment the following
	# plot_animals(filename="bootstrapped_agency_patiency_scores_animals_nondomesticated_animals.npy", outfile_name="appendix_results_domesticated_animals_vs_wild_animals_window_size_0.pdf", window_size=0)



