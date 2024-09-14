import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from scipy.stats import spearmanr


def calculate_average_metrics(file):
    models = ['Llama-2-7b-chat-hf', 'Mistral-7B-Instruct-v0.3', 'Phi-3-mini-128k-instruct', 'Saul-7B-Instruct-v1',
              'Meta-Llama-3.1-8B-Instruct', 'gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09']

    results = dict()
    for model in models:
        data_file = pd.read_excel(file, sheet_name=model)
        mean_fluency = data_file['Fluency'].mean()
        mean_acc = data_file['Accuracy'].mean()
        results[model] = {'fluency': mean_fluency, 'accuracy': mean_acc}
    return results


if __name__ == '__main__':
    file_wo_legal_area = ['evaluation/human/reasons_A_SYA.xlsx', 'evaluation/human/reasons_A_TA.xlsx']
    file_with_legal_area = ['evaluation/human/reasons_B_SYA.xlsx', 'evaluation/human/reasons_B_TA.xlsx']
    s_A = calculate_average_metrics('evaluation/human/reasons_A_SYA.xlsx')
    # print(s_A)
    s_B = calculate_average_metrics('evaluation/human/reasons_B_SYA.xlsx')
    # print(s_B)
    # print('========================')
    t_A = calculate_average_metrics('evaluation/human/reasons_A_TA.xlsx')
    # print(t_A)
    t_B = calculate_average_metrics('evaluation/human/reasons_B_TA.xlsx')
    # print(t_B)

    models = ['Llama-2-7b-chat-hf', 'Mistral-7B-Instruct-v0.3', 'Phi-3-mini-128k-instruct', 'Saul-7B-Instruct-v1',
              'Meta-Llama-3.1-8B-Instruct', 'gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09']
    result_list = []

    wo_fluency = []
    w_fluency = []
    wo_accuracy = []
    w_accuracy = []
    for model in models:
        without_result_fluency = round((s_A[model]['fluency'] + t_A[model]['fluency']) / 2, 3)
        without_result_accuracy = round((s_A[model]['accuracy'] + t_A[model]['accuracy']) / 2, 3)

        with_result_fluency = round((s_B[model]['fluency'] + t_B[model]['fluency']) / 2, 3)
        with_result_accuracy = round((s_B[model]['accuracy'] + t_B[model]['accuracy']) / 2, 3)

        # overall_fluency = round((without_result_fluency + with_result_fluency) / 2, 2)
        # overall_accuracy = round((without_result_accuracy + with_result_accuracy) / 2, 2)

        print(
            f"{model} & {without_result_fluency} & {without_result_accuracy} & {with_result_fluency} & {with_result_accuracy} \\\\")
    #     wo_fluency.append(float(with_result_fluency))
    #     w_fluency.append(float(with_result_fluency))
    #     wo_accuracy.append(float(without_result_accuracy))
    #     w_accuracy.append(float(with_result_accuracy))
    #
    # wo_bleu = pd.read_csv('results/default/date_wise_bleu.tsv', sep=' & ')
    # bleu_wo_list = wo_bleu['G'].tolist()
    # wo_rouge = pd.read_csv('results/default/date_wise_rouge.tsv', sep=' & ')
    # rouge_wo_list = wo_rouge['G'].tolist()
    #
    # w_bleu = pd.read_csv('results/tagged/date_wise_bleu.tsv', sep=' & ')
    # bleu_w_list = w_bleu['G'].tolist()
    # w_rouge = pd.read_csv('results/tagged/date_wise_rouge.tsv', sep=' & ')
    # rouge_w_list = w_rouge['G'].tolist()
    #
    # corr_pearson1, _ = pearsonr(bleu_wo_list, wo_fluency)
    # print(f"Pearson correlation BLUE, without legal area, fluency: {round(corr_pearson1,3)}")
    # corr_pearson2, _ = pearsonr(rouge_wo_list, wo_fluency)
    # print(f"Pearson correlation ROUGE, without legal area, fluency: {round(corr_pearson2,3)}")
    #
    # corr_pearson3, _ = pearsonr(bleu_wo_list, wo_accuracy)
    # print(f"Pearson correlation BLUE, without legal area, accuracy: {round(corr_pearson3,3)}")
    # corr_pearson4, _ = pearsonr(rouge_wo_list, wo_accuracy)
    # print(f"Pearson correlation ROUGE, without legal area, accuracy: {round(corr_pearson4,3)}")
    #
    # print("===================== with legal area ===========================")
    #
    # corr_pearson5, _ = pearsonr(bleu_w_list, w_fluency)
    # print(f"Pearson correlation BLUE, with legal area, fluency: {round(corr_pearson5,3)}")
    # corr_pearson6, _ = pearsonr(rouge_w_list, w_fluency)
    # print(f"Pearson correlation ROUGE, with legal area, fluency: {round(corr_pearson6,3)}")
    #
    # corr_pearson7, _ = pearsonr(bleu_w_list, w_accuracy)
    # print(f"Pearson correlation BLUE, with legal area, accuracy: {round(corr_pearson7,3)}")
    # corr_pearson8, _ = pearsonr(rouge_w_list, w_accuracy)
    # print(f"Pearson correlation ROUGE, with legal area, accuracy: {round(corr_pearson8,3)}")
