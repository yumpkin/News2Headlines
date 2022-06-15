import argparse
import torch
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

def run_sentiment_analysis(document: str, model: str=None):
    print("*** Sentiment analysis ***")
    nlp = pipeline(task="sentiment-analysis", model=model)
    result = nlp(document)[0]

    print(f"Document: '{document}':")
    print(f"label: '{result['label']}', with score: {round(result['score'], 4)}")


def run_text_summarization(document: str, model: str=None, max_length=50, do_sample=False):
    print("*** Text summariztion ***")
    text_generator = pipeline(task="text-summariztion", model=model)
    summary_text = text_generator(document, max_length=max_length, do_sample=do_sample)

    print(f"Document: '{document}'")
    print(f"Generated text:\n{summary_text}")

def main():
    parser = argparse.ArgumentParser('Run MNIS prediction')
    parser.add_argument('--task', type=str, help='Requested task. Available options: sentiment, summariztion', default='pipeline')
    parser.add_argument('--document', type=str, help='document text to be used', required=True)
    args = parser.parse_args()

    start_time = datetime.now()
    if args.task == 'sentiment':
        result = run_sentiment_analysis(args.document)
    elif args.task == 'summariztion':
        result = run_text_summarization(args.document, )
    else:
        raise Exception(f"Task not supported: [{args.task}]")
    end_time = datetime.now()
    print(f"Processing time: {end_time - start_time}\n")

if __name__ == "__main__":
    main()
