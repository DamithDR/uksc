import os

import pandas as pd

open_models = ['Llama-2-7b-chat-hf', 'Mistral-7B-Instruct-v0.3', 'Phi-3-mini-128k-instruct', 'Saul-7B-Instruct-v1',
               'Meta-Llama-3.1-8B-Instruct']
gpt_models = ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09']

with pd.ExcelWriter('evaluation/reasons.xlsx', mode='a', engine='openpyxl',
                    if_sheet_exists='replace') as writer:  # need to exist the reasosns excel from openllms outputs
    for model in gpt_models:
        model_reasons = pd.read_excel('outputs/chatgpt_reasons.xlsx', sheet_name=model)
        model_reasons.to_excel(writer, sheet_name=model, index=False)
