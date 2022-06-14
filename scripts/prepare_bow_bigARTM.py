import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import numpy as np
import json
import random
import string
import re
import nltk, razdel
from nltk.corpus import stopwords
from pymystem3 import Mystem

# Download nltk packages used in this example
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

df = pd.read_json('../dataset/enumerated_shuffled_rbc_dataset.json')
df = df.transpose()

# Customize list of stopwords as needed. Here, we append common
# punctuation and contraction artifacts.
with open('../aux/stopwords-ru.txt', 'r') as f:
    ru_stop_words_extensive = f.read().splitlines()
    
punctuations = list(string.punctuation) + ["—", "«", "»", "\n"]
stop_words = list(set(ru_stop_words_extensive + stopwords.words('russian'))) + punctuations

def get_article_sentences(article_text):
    sentences = list()
    for sentence in razdel.sentenize(article_text):
        sentences.append(sentence.text)
    return sentences

def get_article_tokens(article_sentences):
    tokens = list()
    for sentence in article_sentences:
        for token in razdel.tokenize(sentence):
            if token.text not in stop_words:
                tokens.append(token.text.lower())
    return tokens

def get_article_lemmas(article_sentences):
    mystem = Mystem()
    lemmas = list()
    for sentence in article_sentences:
        sentence_lemmas = mystem.lemmatize(sentence.lower())
        sentence_lemmas = [lemma for lemma in sentence_lemmas if lemma not in stop_words\
          and lemma != " "\
          and not lemma.isdigit()
          and lemma.strip() not in punctuations]
    lemmas+=sentence_lemmas
    return set(lemmas)

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r"[^а-яА-Я]", " ", doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    sentences = get_article_sentences(doc)
    # tokenize document
    lemmas = get_article_lemmas(sentences)
    #filter stopwords out of document
    tokens = get_article_tokens(sentences)
    # re-create document from filtered tokens
    doc = ' '.join(lemmas)
    return doc

normalize_corpus = np.vectorize(normalize_document)

def count_vectorize(articles):
    
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=3, 
        max_features=n_features,
        stop_words=stop_words_ru,
        ngram_range=(1, 1))

    token_count_matrix = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names()
        
    vocab = open("./bigARTM/vocab.rbc.txt", "w")
    for feature in features:
        vocab.write(feature+'\n')
    vocab.close()
    
    docword = open("./bigARTM/docword.rbc.txt", "w")
    cx = token_count_matrix.tocoo() #to coordinates
    for docID, wordID, wordCount in zip(cx.row, cx.col, cx.data):
        wordID += 1 # making it unity based to suit with bigARTM
        docID += 1
        docword.write(f"{docID} {wordID} {wordCount}\n")
    docword.close()

corpus = normalize_corpus(list(df['article_text']))
print(len(corpus))

n_features = 1000
n_components = len(set(df.category))
n_top_words = 20 #number of top words to be extracted for a single topic
count_vectorize(articles)