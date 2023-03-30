import json
from gensim import models
from string import punctuation

def load_vectors(mode="glove-wiki-gigaword-300"):
	if mode == "glove-wiki-gigaword-300":
		import gensim.downloader as api
		word_vectors = api.load("glove-wiki-gigaword-300")
	elif mode == "glove-wiki-gigaword-100":
		import gensim.downloader as api
		word_vectors = api.load("glove-wiki-gigaword-100")
	return word_vectors

def generate_wordcloud(words, save_filename="agency.pdf"):
	x, y = [], []
	for word, sim in word_vectors.most_similar(words, topn=1000):
		if len(word) <= 2:
			continue
		flag = False
		for i in punctuation:
			if i in word:
				flag = True
		if not flag:
			x.append(word)
			y.append(sim)
	print (len(x))
	weights = {i:j for i,j in zip(x,y)}
	maincol = randint(0,360)
	

	def colorfunc_darkblue(word=None, font_size=None, 
		  position=None, orientation=None, 
		  font_path=None, random_state=None):   
		color = 255
		return "hsl(%d, %d%%, %d%%)" % (color,randint(65, 75)+font_size / 7, randint(35, 45)-font_size / 10)

	def colorfunc(word=None, font_size=None, 
		  position=None, orientation=None, 
		  font_path=None, random_state=None):   
		color = randint(maincol-10, maincol+10)
		if color < 0:
			color = 360 + color
		return "hsl(%d, %d%%, %d%%)" % (color,randint(65, 75)+font_size / 7, randint(35, 45)-font_size / 10)
	if save_filename == "patiency.pdf":
		# we want patiency to be darkblue
		wordcloud = WordCloud(background_color="white", 
				  ranks_only=False, 
				  max_font_size=120,
				  color_func=colorfunc_darkblue,
				  height=600,width=800).generate_from_frequencies(weights)
	else:
		wordcloud = WordCloud(background_color="white", 
				  ranks_only=False, 
				  max_font_size=120,
				  color_func=colorfunc,
				  height=600,width=800).generate_from_frequencies(weights)
	plt.imshow(wordcloud,interpolation="bilinear")
	plt.axis("off")
	plt.savefig(save_filename, format="pdf")


word_vectors = load_vectors()
agency = ["change", "cause", "aim", "intentional", "harm", "help", "decide", "control", "moral", "plan", "communicate", "think", "choose", "deliberate", "create", "guilty", "responsible", "act", "do", "express"]
patiency = ["pain", "pleasure", "enraged", "afraid", "hunger", "thirsty", "sad", "proud", "embarrassed", "joy", "angry", "happy", "conscious", "aware", "experience", "imagine", "awake", "suffer", "enjoy", "desire"]

from wordcloud import WordCloud
from random import randint
import matplotlib.pyplot as plt

generate_wordcloud(agency, "agency.pdf")
generate_wordcloud(patiency, "patiency.pdf")
