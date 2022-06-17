import rouge_score
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate import bleu
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import numpy as np


def calculate_Rouge_Score(summary, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum', 'rougeL'], use_stemmer=True)
    scores = scorer.score(summary,reference)
    return scores

def getTokens(test):
    test = test.translate(str.maketrans('','',string.punctuation))
    test_list = word_tokenize(test)
    return test_list


def calculate_Bleu_Score(summary, reference):
    summary_tokens = getTokens(summary)
    reference_tokens = getTokens(reference)
    reference_tokens = [reference_tokens]
    smoothie = SmoothingFunction().method1
    weights_unigram = (1, 0, 0)
    weights_bigram = (0, 1, 0)
    weights_trigram = (0, 0, 1)
    bleu_score_unigram = sentence_bleu(reference_tokens,summary_tokens,smoothing_function=smoothie, weights = weights_unigram)
    bleu_score_bigram = sentence_bleu(reference_tokens,summary_tokens,smoothing_function=smoothie, weights = weights_bigram)
    bleu_score_trigram = sentence_bleu(reference_tokens,summary_tokens,smoothing_function=smoothie, weights = weights_trigram)
    return bleu_score_unigram, bleu_score_bigram, bleu_score_trigram


def evaluate(summary_sentence, refrence_sentence):
    #Calculating Bleu Scores
    bleu = np.asarray(calculate_Bleu_Score(summary_sentence, refrence_sentence), dtype=np.float32)
    #Calculating Rouge Scores
    rouge_scores_list = calculate_Rouge_Score(summary_sentence, refrence_sentence)
    rouge1 = np.asarray(rouge_scores_list['rouge1'], dtype = np.float32)
    rouge2 = np.asarray(rouge_scores_list['rouge2'], dtype = np.float32)
    rougeL = np.asarray(rouge_scores_list['rougeL'], dtype = np.float32)
    rougeLsum = np.asarray(rouge_scores_list['rougeLsum'], dtype = np.float32)

    return bleu, rouge1, rouge2, rougeL, rougeLsum


def evalute_sample_from_dataset(summary_data='./outputs/predictions_headline_article_text.csv', num_example_articles=100):
    if isinstance(summary_data, str):
        data = pd.read_csv(summary_data, index_col=0)
    elif isinstance(summary_data, pd.DataFrame):
        data = summary_data

    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    num_examples = min(num_example_articles, len(data.index))
    data = data.values.tolist()


    bleu_scores = np.zeros((1,3), dtype=float)
    rouge1_scores = np.zeros((1,3), dtype=float)
    rouge2_scores = np.zeros((1,3), dtype=float)
    rougeL_scores = np.zeros((1,3), dtype=float)
    rougeLsum_scores = np.zeros((1,3), dtype=float)

    count = 0

    for example in data:
        #Generating Summary
        if count == num_examples:
            break

        summary = example[0]
        reference = example[1]
        #Calculating Bleu Scores
        bleu = np.asarray(calculate_Bleu_Score(summary, reference), dtype=np.float32)
        bleu_scores = np.sum((bleu_scores, bleu), axis = 0) 
        #Calculating Rouge Scores
        rouge_scores_list = calculate_Rouge_Score(summary, reference)
        rouge1 = np.asarray(rouge_scores_list['rouge1'], dtype = np.float32)
        rouge1_scores = np.sum((rouge1_scores, rouge1), axis=0)
        rouge2 = np.asarray(rouge_scores_list['rouge2'], dtype = np.float32)
        rouge2_scores = np.sum((rouge2_scores, rouge2), axis=0)
        rougeL = np.asarray(rouge_scores_list['rougeL'], dtype = np.float32)
        rougeL_scores = np.sum((rougeL_scores, rougeL), axis=0)
        rougeLsum = np.asarray(rouge_scores_list['rougeLsum'], dtype = np.float32)
        rougeLsum_scores = np.sum((rougeLsum_scores, rougeLsum), axis=0)
        count+=1
        # print("Example Number {} done".format(count))
        
    bleu_scores /= num_examples
    rouge1_scores/=num_examples
    rouge2_scores/=num_examples
    rougeL_scores/=num_examples
    rougeLsum_scores/=num_examples

    print("Final Blue Scores", bleu_scores)
    print("Final ROUGE1 Scores", rouge1_scores)
    print("Final ROUGE2 Scores", rouge2_scores)
    print("Final ROUGEL Scores", rougeL_scores)
    print("Final ROUGELsum Scores", rougeLsum_scores)
    return bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, rougeLsum_scores