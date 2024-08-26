import numpy as np
import pandas as pd
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score

from util.eval import eval_decisions
from util.setting import tag


def evaluate_decisions(model_name, input_path, tag_path, filter_date=None):
    decisions = pd.read_excel(input_path, sheet_name=model_name)
    if filter_date:
        decisions = decisions[decisions['date'] > filter_date]
    w_recall, w_precision, w_f1, m_f1 = eval_decisions(decisions, 'predictions', 'gold')
    with open(f'results/{tag_path}decision_stats.tsv', 'a') as f:
        f.write(
            f'{model_name}\t{filter_date}\t{w_recall:.2f}\t{w_precision:.2f}\t{w_f1:.2f}\t{m_f1:.2f}\n')
    return f'{m_f1:.2f}', f'{w_f1:.2f}'


def evaluate_reasons(model_name, input_path, filter_date=None):
    reasons = pd.read_excel(input_path, sheet_name=model_name)
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
    references = reasons['gold'].tolist()
    candidates = reasons['predictions'].tolist()
    P, R, F1 = score(candidates, references, lang='en', verbose=True)
    # F1 =0 # todo remove
    print(
        f'model : {model_name} | filter date {filter_date} | bleu : {np.mean(blue_scores)} | rough : {np.mean(r_scores)} | bert score : {F1}')
    return np.mean(blue_scores), np.mean(r_scores), F1


if __name__ == '__main__':

    if tag:
        tag_path = 'tagged/'
        input_decisions = 'evaluation/decisions_tag.xlsx'
        input_reasons = 'evaluation/reasons_tag.xlsx'
    else:
        tag_path = 'default/'
        input_decisions = 'evaluation/decisions.xlsx'
        input_reasons = 'evaluation/reasons.xlsx'

    models = ['Llama-2-7b-chat-hf', 'Mistral-7B-Instruct-v0.3', 'Phi-3-mini-128k-instruct', 'Saul-7B-Instruct-v1',
              'Meta-Llama-3.1-8B-Instruct', 'gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09']

    global_cutoff_date = '12/31/2023'

    model_cutoff_dates = ['7/31/2023', '10/31/2023', '10/31/2023', '2/28/2023', '12/31/2023', '9/30/2021', '12/31/2023']

    for model, date in zip(models, model_cutoff_dates):
        m_f1_all, w_f1_all = evaluate_decisions(model, input_path=input_decisions, tag_path=tag_path)
        m_f1_model_specific, w_f1_model_specific = evaluate_decisions(model, input_path=input_decisions,
                                                                      tag_path=tag_path, filter_date=date)
        m_f1_global, w_f1_global = evaluate_decisions(model, input_path=input_decisions,
                                                      tag_path=tag_path, filter_date=date)

        with open(f'results/{tag_path}date_wise_macro_f1.tsv', 'a') as f:
            f.write(f'{model}\t{m_f1_all}\t{m_f1_model_specific}\t{m_f1_global}\n')

        with open(f'results/{tag_path}date_wise_weighted_f1.tsv', 'a') as f:
            f.write(f'{model}\t{w_f1_all}\t{w_f1_model_specific}\t{w_f1_global}\n')

        bleu_all, rouge_all, bertF1_all = evaluate_reasons(model, input_path=input_reasons)
        bleu_model, rouge_model, bertF1_model = evaluate_reasons(model, input_path=input_reasons, filter_date=date)
        bleu_global, rouge_global, bertF1_global = evaluate_reasons(model, input_path=input_reasons, filter_date=date)

        with open(f'results/{tag_path}date_wise_bleu.tsv', 'a') as f:
            f.write(f'{model}\t{bleu_all}\t{bleu_model}\t{bleu_global}\n')

        with open(f'results/{tag_path}date_wise_rouge.tsv', 'a') as f:
            f.write(f'{model}\t{rouge_all}\t{rouge_model}\t{rouge_global}\n')

        with open(f'results/{tag_path}date_wise_bertscore.tsv', 'a') as f:
            f.write(f'{model}\t{bertF1_all}\t{bertF1_model}\t{bertF1_global}\n')
