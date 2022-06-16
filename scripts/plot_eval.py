import matplotlib.pyplot as plt
import numpy as np

def plot_rouge(rouge_scores_array):

    column_headers = ["Precision", "Recall", "F-Measure"]
    row_headers = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum"]
    cell_text_rouge = rouge_scores_array
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
    plt.rcParams["figure.figsize"] = [7.00, 10.0]
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(1, 1)
    data = cell_text_rouge
    columns = column_headers
    rows = row_headers
    axs.axis('tight')
    axs.axis('off')
    the_table = axs.table(cellText=data, 
                        colLabels=columns,
                        rowLabels=rows,
                        rowColours = rcolors,
                        colColours = ccolors,
                        loc='center')
    plt.title('ROUGE SCORE EVALUATION')
    plt.show()



def plot_bleu(bleu_scores_array):

    column_headers = ["Bleu Scores"]
    row_headers = ["BLEU-1", "BLEU-2", "BLEU-3"]
    cell_text_bleu = bleu_scores_array.T
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
    plt.rcParams["figure.figsize"] = [7.00, 8.00]
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(1, 1)
    axs.axis('tight')
    axs.axis('off')
    table = plt.table(cellText=cell_text_bleu,
                        rowLabels=row_headers,
                        rowColours=rcolors,
                        colColours = ccolors,
                        colLabels=column_headers,
                        loc='center')
    plt.title('BLEU SCORE EVALUATION')
    plt.show()