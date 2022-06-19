# Text Summarization of Russian news articles

## Introduction

### Project overview

The project aimed to invistigate extractive and abstractive methodoligies of summarization, generated summaries have been evaluated with `ROUGE` and `BlEU` metrics.
### Dataset:

The dataset of use, is a set of РБС news articles. it is rather compact ~2000 articles. It is used for fine-tunning `T5` transformer model, and will be divided into `train`, `test`, `evaluate` portions.

### Model:
The model of use if `T5` since it deals with each NLP task in `text-to-text` manner, which gives it a generic usage and makes it suitable for summarization.

### Folders

The project is structured into the following folders:

- ***`dataset`*** contains a bit over `2000` РБС articles in the format of `JSON` and `txt` files. Alongside with the files `gazeta_2k.json` and `ria_2k.json` each of which contains `2000` articles of Gazeta and `RIA` respectively.
`gazeta` and `ria` files are just a pre-proccessed slices from two large opensource datasets `Russiya Segodnyia` and `Gazeta v.1`. Credits to them for offering it opensourced.
- - `JSON` files are well structred, and have two variants:
    - Each article is indexed by a hash of its text body, for further possible checkups.
    - Each article is indexed by just by its enumerated index (its postional order in the dataset)


- ***`notebooks`*** contains the notebooks:
    
  -  `summarization_finetune.ipynb` contains the main task of the project, splitting the dataset into `train, test, validate`, load it and finetune the model, generate abstractive summaries, evaluate the results.
  
  - `baseline.ipynb` dedicated for getting the baselines and evaluting them along with the abstract summaries.
  
  - `dataset_stats.ipynb` for all comparison parameters between the datasets used.
    
  -  `extractive_summarization.ipynb` covers various techniques of extractive summarization, from `tf-idf` to `BERT-extractive`.
  
  -  `bag_of_words.ipynb` for creating a Bag of Words(BoW) model in the `UCI` format that is needed later for `bigARTM` thematic modelling.
  
  -  `clustering_of_article.ipynb` is for creating `k-means` clusters, with various experimentation and results analysis.
  
  -  `text_analysis.ipynb` general standard measurments of linguistic interset, such as text lingual statistics, and frequency distributions of tokens.
  
  -  `data_preparation.ipynb` the creation of thed dataset in an interactive form. contains parsing and data-structuring functions.

- ***`scripts`*** folder that contains utility scripts that are more convenient to be used in a command-line interface (CLI).
  - `bow_uci.py` a script that can be used to build the bag of words for `bigARTM`.
  - `evalute.py` is used to evaluate summaries, it has functions that take a `csv` file as for all `summary-article` pairs, and then print the average scores of `ROUGE` and `BLEU` or just `summary-sentence` and `source-sentence` as an argument, and then returns its corresponding scores.
  - `plot_eval.py` is for plotting the results of evaluation into pretty tables.
  - `kmeans.py` is used for clustering a set of articles.
  - `parse_rbc.py` a script that runs to parse RBC website and append articles to dataset. or create a one if does not exist.
  
- ***`outputs`*** contains the outputs of scripts/notebooks, like `csv` files they might generate.
  
- ***`Docker`*** contains needed files for containering the project, with a script entry to a `CLI` interface. *(In progress)*.

- ***`aux`*** for auxiliary files, such as russian extensive stopwords file.
