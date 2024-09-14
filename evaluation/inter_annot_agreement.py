import pandas as pd
from scipy.stats import pearsonr


def run():
    sya_files = ['evaluation/human/reasons_A_SYA.xlsx', 'evaluation/human/reasons_B_SYA.xlsx']
    ta_files = ['evaluation/human/reasons_A_TA.xlsx', 'evaluation/human/reasons_B_TA.xlsx']

    models = ['Llama-2-7b-chat-hf', 'Mistral-7B-Instruct-v0.3', 'Phi-3-mini-128k-instruct', 'Saul-7B-Instruct-v1',
              'Meta-Llama-3.1-8B-Instruct', 'gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09']

    fluency_SYA = []
    accuracy_SYA = []
    fluency_TA = []
    accuracy_TA = []

    for file in sya_files:
        for model in models:
            df = pd.read_excel(file, sheet_name=model)
            fluency_SYA.append(df['Fluency'].tolist())
            accuracy_SYA.append(df['Accuracy'].tolist())

    for file in ta_files:
        for model in models:
            df = pd.read_excel(file, sheet_name=model)
            fluency_TA.append(df['Fluency'].tolist())
            accuracy_TA.append(df['Accuracy'].tolist())

    fluency_SYA = [value for lst in fluency_SYA for value in lst]
    accuracy_SYA = [value for lst in accuracy_SYA for value in lst]
    fluency_TA = [value for lst in fluency_TA for value in lst]
    accuracy_TA = [value for lst in accuracy_TA for value in lst]

    accuracy_kappa_score,_ = pearsonr(accuracy_SYA, accuracy_TA)
    fluency_kappa_score,_ = pearsonr(fluency_SYA, fluency_TA)

    print(f'Fluency : {fluency_kappa_score:.3f} | Accuracy : {accuracy_kappa_score:.3f}')


if __name__ == '__main__':
    run()
