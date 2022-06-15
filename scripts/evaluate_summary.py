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
    #Rouge N: N-gram scoring
    #Rouge L: sentence-level: Compute longest common subsequence (LCS) between two pieces of text. Newlines are ignored.
    #RougeLsum: summary-level: Newlines in the text are interpreted as sentence boundaries, and the LCS is computed between each pair of reference and candidate sentences, and something called union-LCS is computed.
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
    weights_unigram = (1, 0, 0, 0)
    weights_bigram = (0, 1, 0, 0)
    weights_trigram = (0, 0, 1, 0)
    weights_fourgram = (0, 0, 0, 1)
    bleu_score_unigram = sentence_bleu(reference_tokens,summary_tokens,smoothing_function=smoothie, weights = weights_unigram) #BLEU-1,2,3,4
    bleu_score_bigram = sentence_bleu(reference_tokens,summary_tokens,smoothing_function=smoothie, weights = weights_bigram)
    bleu_score_trigram = sentence_bleu(reference_tokens,summary_tokens,smoothing_function=smoothie, weights = weights_trigram)
    bleu_score_fourgram = sentence_bleu(reference_tokens,summary_tokens,smoothing_function=smoothie, weights = weights_fourgram)
    return bleu_score_unigram, bleu_score_bigram, bleu_score_trigram, bleu_score_fourgram

def evalute_sample_from_dataset(data_csv_path='../outputs/unique_predictions_headline_article_text.csv', num_example_articles=100):
    unique_headline_summaries = pd.read_csv(data_csv_path, index_col=0)
    dataset_size = unique_headline_summaries.size
    data = unique_headline_summaries.values.tolist()

    num_examples = num_example_articles
    count = 0

    for example in data:
        #Generating Summary
        if count == num_examples:
            break

        summary = example[0]
        reference = example[1]
    
        bleu_scores = np.zeros((1,4), dtype=float) #Blue -1,2,3,4
        rouge1_scores = np.zeros((1,3), dtype=float) #Rouge F-measures for rouge 1, rouge 2, rouge L, rouge Lsum
        rouge2_scores = np.zeros((1,3), dtype=float)
        rougeL_scores = np.zeros((1,3), dtype=float)
        rougeLsum_scores = np.zeros((1,3), dtype=float)

        bleu, rouge1, rouge2, rougeL, rougeLsum = evaluate(summary, reference)

        bleu_scores = np.sum((bleu_scores, bleu), axis = 0) 
        rouge1_scores = np.sum((rouge1_scores, rouge1), axis=0)
        rouge2_scores = np.sum((rouge2_scores, rouge2), axis=0)
        rougeL_scores = np.sum((rougeL_scores, rougeL), axis=0)
        rougeLsum_scores = np.sum((rougeLsum_scores, rougeLsum), axis=0)
        bleu_scores /= num_examples
        rouge1_scores/=num_examples
        rouge2_scores/=num_examples
        rougeL_scores/=num_examples
        rougeLsum_scores/=num_examples
        # print("Example Number {} done".format(count))
        count+=1

    print("Final Blue Scores", bleu_scores)
    print("Final ROUGE1 Scores", rouge1_scores)
    print("Final ROUGE2 Scores", rouge2_scores)
    print("Final ROUGEL Scores", rougeL_scores)
    print("Final ROUGELsum Scores", rougeLsum_scores)
    
    return bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, rougeLsum_scores

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