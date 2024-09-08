import os

import pandas as pd

def run(filename):
    models = ['Llama-2-7b-chat-hf', 'Mistral-7B-Instruct-v0.3', 'Phi-3-mini-128k-instruct', 'Saul-7B-Instruct-v1',
              'Meta-Llama-3.1-8B-Instruct', 'gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09']

    global_cutoff_date = '12/31/2023'

    for model in models:
        reasons = pd.read_excel(f'evaluation/{filename}', sheet_name=model)
        reasons = reasons[reasons['date'] > global_cutoff_date]

        if not os.path.exists(f"outputs/filtered_results/{filename}"):
            reasons.to_excel(f"outputs/filtered_results/{filename}", sheet_name=f"{model}", index=False)
        else:
            with pd.ExcelWriter(f'outputs/filtered_results/{filename}', mode='a', engine='openpyxl',
                                if_sheet_exists='replace') as writer:
                reasons.to_excel(writer, sheet_name=f"{model}", index=False)


if __name__ == '__main__':
    file_name = 'reasons_tag.xlsx' # change
    run(file_name)