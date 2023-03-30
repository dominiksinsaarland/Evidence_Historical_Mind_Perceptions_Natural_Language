#import fasttext
import os
import sys

os.makedirs("models_word2vec", exist_ok=True)
year= sys.argv[1]
# i are bootstrap samples, e.g. 1-9
try:
	i = sys.argv[2]
except:
	i = None
#   #python train_model_word2vec.py coha/$year/"bootstrapped"$i"_"$year.txt $i

if i is not None:
	path = "coha/" + year + "/bootstrapped" + i + "_" + year+ ".txt"
else:
	path = "coha/" + year + "/all_" + year + ".txt"

if i is not None:
	outpath = "models_word2vec_" + str(i)
else:
	outpath = "models_word2vec"

os.makedirs(outpath, exist_ok=True)
from gensim.models.word2vec import LineSentence
sentences = LineSentence(path)

from gensim.models import Word2Vec
model = Word2Vec(sentences=sentences, vector_size=100, window=8, min_count=10, workers=4)
out_model = path.split("_")[-1].split(".")[0]
model.save(outpath + "/" + year + ".model")


