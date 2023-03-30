# What is (and was) a Person? Evidence on Historical Mind Perceptions from Natural Language

This github repo contains the reproduction package for our paper about [evidence on historical mind perceptions from natural language](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3959847).  

## Data

In this paper, we conduct a lare survey where we ask survey participants to rank 255 entities along different axis (e.g., perceived agency and experience). This survey can be found in 

```shell
data/items-agg.csv
```

## Installation

Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/) and linux, the environment can be installed with the following command:
```shell
conda create -n agency_experience python=3.6
conda activate agency_experience
pip install -r requirements.txt
```

## setup

Extract vectors: this simply extracts word embeddings for all words of interest from various pre-trained embeddings, such that we do not have to load all embeddings all the time (have a look at the python file for more detail and from where to download some word embeddings).

```shell
python src/extract_word_vectors.py
```

## main results (scatterplots showing correlation between word embeddings and survey results)

to create the scatterplots in our Figure 1 in the paper, run 

```shell
python create_scatterplots.py
```

## main results (historical analysis)

To create our historical analysis, i.e., Figure 2 in the paper, run the following steps:

- Download our pre-computed and aligned word embeddings using COHA and word2vec [here](https://www.dropbox.com/s/7eaiwxhq6017g24/aligned_historical_word2vec_models.zip?dl=0) and extract in the same directory
- Extract scores for all entities of interest, and plot afterwards.
- To reproduce the trianing of word embeddings on COHA, see the last section in this github repo.

```shell
python src/get_scores_historical_analysis.py
```

```shell
python src/plot_historical_analysis.py
```

## Appendix Results

### Appendix B
create word clouds for figure "Semantic Poles for Agency and Patiency"

```shell
python appendix-src/most_similar_agency_patiency_words.py 
```

Generate histograms, top and bottom words, correlation language measure and difference agency-patiency

```shell
python appendix-src/appendix_B_language_measures.py
```
### Appendix C
```shell
python appendix-src/appendix_C_survey_measures.py
```

### Appendix D

```shell
python appendix-src/appendix_D_survey_language.py
```

## Reproduce training word embeddings

This is rather cumbersome, these steps have to be followed

- obtain a license from [COHA](https://www.english-corpora.org/coha/)
- download the COHA corpus
- train own word embeddings on COHA 10 times (for bootstrapping)
- align the embeddings, see [Hamilton et al., 2016](https://aclanthology.org/P16-1141/)
- extract the perceived agency and experience scores from COHA
- create a nice plot

First, we need to preprocess COHA. COHA is organized in folders (e.g., 1820/), and each folder contains text from various sources. We simply concatenate all these texts (using the unix "cat" tool) for each decade into a file year + "all_" + year + ".txt", e.g., 1820/all_1820.txt 

Next, we train 10 different word embeddings models for all decades, using different bootstrapped variants of the corpus (this takes a while to train). The following line will compute models for the fifth bootstrapping run. Run the script ten times with numbers 0-9.

```shell
bash scripts/train_models_word2vec_bootstrap.sh 5
```

For alignment, run

```shell
python coha-code/align_embeddings_all.py
```

And this yields the same aligned models as the one in the zip file (there might be small differences in reproductino because we did not set a seed to bootstrap the corpora, and we did not set a seed while running word2vec).


