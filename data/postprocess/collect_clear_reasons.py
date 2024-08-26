import os
import shutil

import pandas as pd

from util.setting import tag


def run(input, original, output):
    gpt_models = ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09']

    shutil.copy(original, output)
    with pd.ExcelWriter(output, mode='a', engine='openpyxl',
                        if_sheet_exists='replace') as writer:  # need to exist the reasosns excel from openllms outputs
        for model in gpt_models:
            model_reasons = pd.read_excel(input, sheet_name=model)
            model_reasons.to_excel(writer, sheet_name=model, index=False)


if __name__ == '__main__':

    if tag:
        input = 'outputs/chatgpt_reasons_tag.xlsx'
        original = 'outputs/reasons_tag.xlsx'
        output = 'evaluation/reasons_tag.xlsx'
    else:
        input = 'outputs/chatgpt_reasons.xlsx'
        original = 'outputs/reasons.xlsx'
        output = 'evaluation/reasons.xlsx'
    run(input, original, output)
