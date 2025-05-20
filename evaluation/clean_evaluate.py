import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from util.eval import eval_decisions


def eval_classifications(input_file, sheet_name, alias):
    model_decisions = pd.read_excel(input_file, sheet_name=sheet_name)
    decisions = model_decisions['predictions']
    decisions = list(map(lambda x: x.lower(), decisions))
    decisions = list(map(lambda x: x.replace('allow.', 'allow'), decisions))
    decisions = list(map(lambda x: x.replace('dismiss.', 'dismiss'), decisions))
    decisions = list(map(lambda x: x.replace('\n', ''), decisions))
    decisions = list(map(lambda x: x.replace('<<sys>>', ''), decisions))
    decisions = list(map(lambda x: x.replace('<</sys>>', ''), decisions))
    decisions = list(map(lambda x: x.replace('[', ''), decisions))
    decisions = list(map(lambda x: x.replace(']', ''), decisions))
    model_decisions['predictions'] = decisions

    decisions = pd.read_excel(input_file, sheet_name=sheet_name)
    w_recall, w_precision, w_f1, m_f1 = eval_decisions(decisions, 'predictions', 'gold')
    with open(f'results/default/new_date_wise_macro_f1.csv', 'a') as f:
        f.write(f'{alias} & {m_f1:.3f} & na & na\n')

    with open(f'results/default/new_date_wise_weighted_f1.csv', 'a') as f:
        f.write(f'{alias} & {w_f1:.3f} & na & na\n')


def eval_reasons(input_file, sheet_name, alias):
    reasons = pd.read_excel(input_file, sheet_name=sheet_name)
    reasons = reasons.fillna('empty')
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
    print(
        f'model : {alias} | bleu : {np.mean(blue_scores)} | rough : {np.mean(r_scores)}')
    bleu_all, rouge_all = np.mean(blue_scores), np.mean(r_scores)
    with open(f'results/default/new_date_wise_bleu.tsv', 'a') as f:
        f.write(f'{alias} & {bleu_all:.3f} & 0 & 0\n')

    with open(f'results/default/new_date_wise_rouge.tsv', 'a') as f:
        f.write(f'{alias} & {rouge_all:.3f} & 0 & 0\n')


if __name__ == '__main__':
    # deep seek
    # eval_classifications('outputs/deepseek_decisions_background.xlsx','deepseek-chat','deepseek_normal_background')
    # eval_reasons('outputs/deepseek_reasons_background.xlsx','deepseek-chat','deepseek_normal_background')

    # deep seek
    # eval_classifications('outputs/deepseek_decisions_judgment.xlsx', 'deepseek-chat', 'deepseek_normal_judgment')
    # eval_reasons('outputs/deepseek_reasons_judgment.xlsx', 'deepseek-chat', 'deepseek_normal_judgment')

    # deep seek
    # eval_classifications('outputs/rag_deepseek_decisions_background.xlsx','deepseek-chat','rag_deepseek_normal_background')
    # eval_reasons('outputs/rag_deepseek_reasons_background.xlsx','deepseek-chat','rag_deepseek_normal_background')

    # deep seek rag
    # eval_classifications('outputs/rag_deepseek_decisions_judgment.xlsx', 'deepseek-chat', 'rag_deepseek_normal_judgment')
    # eval_reasons('outputs/rag_deepseek_reasons_judgment.xlsx', 'deepseek-chat', 'rag_deepseek_normal_judgment')

    # deep seek random rag
    # eval_classifications('outputs/rag_random_deepseek_decisions_background.xlsx','deepseek-chat','rag_random_deepseek_normal_background')
    # eval_reasons('outputs/rag_random_deepseek_reasons_background.xlsx','deepseek-chat','rag_random_deepseek_normal_background')

    # deep seek random rag
    # eval_classifications('outputs/rag_random_deepseek_decisions_judgment.xlsx', 'deepseek-chat',
    #                      'rag_random_deepseek_normal_judgment')
    # eval_reasons('outputs/rag_random_deepseek_reasons_judgment.xlsx', 'deepseek-chat', 'rag_random_deepseek_normal_judgment')

    #chatgpt random
    # eval_classifications('outputs/rag_random_chatgpt_decisions_judgment.xlsx', 'gpt-3.5-turbo-0125',
    #              'rag_random_gpt3.5_normal_background')
    # eval_reasons('outputs/rag_random_chatgpt_reasons_background.xlsx', 'gpt-3.5-turbo-0125',
    #              'rag_random_gpt3.5_normal_background')
    #
    # eval_classifications('outputs/rag_random_chatgpt_decisions_judgment.xlsx', 'gpt-3.5-turbo-0125',
    #                      'rag_random_gpt3.5_normal_judgment')
    # eval_reasons('outputs/rag_random_chatgpt_reasons_judgment.xlsx', 'gpt-3.5-turbo-0125',
    #              'rag_random_gpt3.5_normal_judgment')

    # eval_classifications('outputs/rag_random_chatgpt_decisions_judgment.xlsx', 'gpt-4-turbo-2024-04-09',
    #                      'rag_random_gpt4_normal_background')
    # eval_reasons('outputs/rag_random_chatgpt_reasons_background.xlsx', 'gpt-4-turbo-2024-04-09',
    #              'rag_random_gpt4_normal_background')
    #
    # eval_classifications('outputs/rag_random_chatgpt_decisions_judgment.xlsx', 'gpt-4-turbo-2024-04-09',
    #                      'rag_random_gpt4_normal_judgment')
    # eval_reasons('outputs/rag_random_chatgpt_reasons_judgment.xlsx', 'gpt-4-turbo-2024-04-09',
    #              'rag_random_gpt4_normal_judgment')
    eval_reasons('outputs/lora_flan_t5_inference.xlsx', 'data',
                 'loraflant5')
