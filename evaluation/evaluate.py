import numpy as np
import pandas as pd
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

from util.eval import eval_decisions


def evaluate_decisions(model_name, filter_date=None):
    decisions = pd.read_excel('evaluation/decisions.xlsx', sheet_name=model_name)
    if filter_date:
        decisions = decisions[decisions['date'] > filter_date]
    w_recall, w_precision, w_f1, m_f1 = eval_decisions(decisions, 'predictions', 'gold')
    with open(f'evaluation/decision_stats.tsv', 'a') as f:
        f.write(
            f'{model_name}\t{w_recall:.2f}\t{w_precision:.2f}\t{w_f1:.2f}\t{m_f1:.2f}\n')


def evaluate_reasons(model_name, filter_date=None):
    reasons = pd.read_excel('evaluation/reasons.xlsx', sheet_name=model_name)
    reasons = reasons.fillna('empty')
    if filter_date:
        reasons = reasons[reasons['date'] > filter_date]
    blue_scores = []
    r_scores = []
    for reference_paragraph, candidate_paragraph in zip(reasons['gold'], reasons['predictions']):
        # Tokenize the entire paragraph into words
        reference_chars = list(reference_paragraph)
        candidate_chars = list(candidate_paragraph)

        # Calculate corpus-level BLEU-4
        bleu_score = sentence_bleu([reference_chars], candidate_chars, weights=(0.25, 0.25, 0.25, 0.25),
                                   smoothing_function=SmoothingFunction().method1)
        blue_scores.append(bleu_score)
        # print(f"Corpus-level BLEU-4 Score: {bleu_score:.4f}")

        # Initialize the scorer for ROUGE-2
        scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
        scores = scorer.score(reference_paragraph, candidate_paragraph)

        # Extract and print the ROUGE-2 score
        rouge_2_score = scores['rouge2'].fmeasure
        r_scores.append(rouge_2_score)
        # print(f"ROUGE-2 Score for the Paragraph: {rouge_2_score:.4f}")
    print(f'model : {model_name} | bleu : {np.mean(blue_scores)} | rough : {np.mean(r_scores)}')

if __name__ == '__main__':
    models = ['Llama-2-7b-chat-hf', 'Mistral-7B-Instruct-v0.3', 'Phi-3-mini-128k-instruct', 'Saul-7B-Instruct-v1',
              'Meta-Llama-3.1-8B-Instruct', 'gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09']
    for model in models:
        evaluate_reasons(model)
