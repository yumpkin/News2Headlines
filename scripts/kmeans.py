import nltk, razdel
from nltk.corpus import stopwords
# Download nltk packages used in this example
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from pymystem3 import Mystem
import string, regex as re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

from sklearn.cluster import KMeans

def get_article_sentences(article_text):
    sentences = list()
    for sentence in razdel.sentenize(article_text):
        sentences.append(sentence.text)
    return sentences

def get_article_tokens(article_sentences, stop_words):
    tokens = list()

    for sentence in article_sentences:
        for token in razdel.tokenize(sentence):
            if token.text not in stop_words:
                tokens.append(token.text.lower())
    return tokens

def get_article_lemmas(article_sentences, stop_words):
    mystem = Mystem()
    lemmas = list()
    
    for sentence in article_sentences:
        sentence_lemmas = mystem.lemmatize(sentence.lower())
        sentence_lemmas = [lemma for lemma in sentence_lemmas if lemma not in stop_words\
          and lemma != " "\
          and lemma.strip() not in punctuations]
    lemmas+=sentence_lemmas
    return set(lemmas)

def normalize_document(doc, stop_words):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r"[^а-яА-Я]", " ", doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    sentences = get_article_sentences(doc)
    # tokenize document
    lemmas = get_article_lemmas(sentences, stop_words)
    #filter stopwords out of document
    tokens = get_article_tokens(sentences, stop_words)
    # re-create document from filtered tokens
    doc = ' '.join(lemmas)
    return doc



def calculate_cv_matrix(corpus, gram_count, min_df=0.5, max_df=1):
    ngrams = {'only_unigrams':(1,1), 'unigrams_bigrams':(1,2), 'only_bigrams':(2,2), 'bigrams_trigrams':(2,3), 'only_trigrams':(3,3)}
    cv = CountVectorizer(ngram_range=ngrams[gram_count], min_df=min_df, max_df=max_df, stop_words=stop_words)
    cv_matrix = cv.fit_transform(corpus)
    cv_matrix.shape
    return cv_matrix, cv


def apply_kmeans(cv_matrix, cluster_count):
    NUM_CLUSTERS = cluster_count
    km = KMeans(n_clusters=NUM_CLUSTERS, max_iter=10000, n_init=100, random_state=32).fit(cv_matrix)
    return km


def print_clusters(km, cv, df, NUM_CLUSTERS):
    df['kmeans_cluster'] = km.labels_
    article_clusters = (df[['category', 'kmeans_cluster',]]
                  .sort_values(by=['kmeans_cluster'], 
                               ascending=False)
                  .groupby('kmeans_cluster').head(20))
    article_clusters = article_clusters.copy(deep=True)
    
    feature_names = cv.get_feature_names()
    topn_features = 15
    ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]
    counts = dict()

    # get key features for each cluster
    # get articles belonging to each cluster
    for cluster_num in range(NUM_CLUSTERS):
        key_features = [feature_names[index] 
                            for index in ordered_centroids[cluster_num, :topn_features]]
        articles = article_clusters[article_clusters['kmeans_cluster'] == cluster_num]['category'].values.tolist()
        for category in articles:
            counts[category] = counts.get(category, 0) + 1

        counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
        print('CLUSTER #'+str(cluster_num+1))
        print('Key Features:', key_features, "\n")
        print('Corresponding categories: \n', counts)
        print('-'*80)

def main(dataset_file = "../dataset/enumerated_shuffled_rbc_dataset.json", stopwords_file = '../aux/stopwords-ru.txt'):
# Customize list of stopwords as needed. Here, we append common
# punctuation and contraction artifacts.
    with open(stopwords_file, 'r') as f:
        ru_stop_words_extensive = f.read().splitlines()
        
    punctuations = list(string.punctuation) + ["—", "«", "»", "\n"]
    stop_words = list(set(ru_stop_words_extensive + stopwords.words('russian'))) + punctuations


    df = pd.read_json(dataset_file)
    df = df.transpose()

    normalize_corpus = np.vectorize(normalize_document)
    norm_corpus = normalize_corpus(list(df['article_text']))
    cv_matrix, cv = calculate_cv_matrix(norm_corpus)
    km = apply_kmeans(cv_matrix, 6)
    Counter(km.labels_)
    print_clusters(km, cv, df, 6)
    return cv, km