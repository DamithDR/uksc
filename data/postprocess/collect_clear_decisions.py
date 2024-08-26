import os

import pandas as pd

from util.setting import tag

open_models = ['Llama-2-7b-chat-hf', 'Mistral-7B-Instruct-v0.3', 'Phi-3-mini-128k-instruct', 'Saul-7B-Instruct-v1',
               'Meta-Llama-3.1-8B-Instruct']
gpt_models = ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09']


def run(input_file, input_gpt_file, output_file):
    # clean and clear the decisions
    for model in open_models:
        model_decisions = pd.read_excel(input_file, sheet_name=model)
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
        if not os.path.exists("evaluation/decisions.xlsx"):
            model_decisions.to_excel(output_file, sheet_name=model, index=False)
        else:
            with pd.ExcelWriter(output_file, mode='a', engine='openpyxl',
                                if_sheet_exists='replace') as writer:
                model_decisions.to_excel(writer, sheet_name=model, index=False)

    for model in gpt_models:
        model_decisions = pd.read_excel(input_gpt_file, sheet_name=model)
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
        if not os.path.exists(output_file):
            model_decisions.to_excel(output_file, sheet_name=model, index=False)
        else:
            with pd.ExcelWriter(output_file, mode='a', engine='openpyxl',
                                if_sheet_exists='replace') as writer:
                model_decisions.to_excel(writer, sheet_name=model, index=False)


if __name__ == '__main__':

    if tag:
        input_file = 'outputs/decisions.xlsx'
        input_gpt_file = 'outputs/chatgpt_decisions.xlsx'
        output_file = 'evaluation/decisions_tag.xlsx'
    else:
        input_file = 'outputs/decisions.xlsx'
        input_gpt_file = 'outputs/chatgpt_decisions.xlsx'
        output_file = 'evaluation/decisions.xlsx'




    run(input_file, input_gpt_file, output_file)
