import json
import os
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from experiments.open_llms import get_messages_for_labels, get_messages_for_reasoning
from util.eval import eval_decisions


def run(input_column):
    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    model = 'deepseek-chat'
    df = pd.read_excel('data/test_data_extended.xlsx', sheet_name='data')
    decision_messages = get_messages_for_labels(df, 'default', input_column)
    decisions = []
    for messages in tqdm(decision_messages, total=len(decision_messages)):
        response = client.chat.completions.create(
            messages=messages,
            stream=False,
            model=model,
            temperature=0.1,
            max_tokens=2048,
        )
        decisions.append(response.choices[0].message.content)
        time.sleep(0.1)

    decisions_df = pd.DataFrame()
    decisions_df['date'] = df['decision_date']
    decisions_df['gold'] = df['decision_label']
    decisions_df['predictions'] = decisions

    if not os.path.exists(f"outputs/deepseek_decisions_{input_column}.xlsx"):
        decisions_df.to_excel(f"outputs/deepseek_decisions_{input_column}.xlsx", sheet_name=model, index=False)
    else:
        with pd.ExcelWriter(f'outputs/deepseek_decisions_{input_column}.xlsx', mode='a', engine='openpyxl',
                            if_sheet_exists='replace') as writer:
            decisions_df.to_excel(writer, sheet_name=model, index=False)

    w_recall, w_precision, w_f1, m_f1 = eval_decisions(decisions_df, 'predictions', 'gold')
    with open(f'decision_stats_{input_column}.tsv', 'a') as f:
        f.write(
            f'{model}\t{w_recall}\t{w_precision}\t{w_f1}\t{m_f1}\n')

    # reasoning
    reason_messages = get_messages_for_reasoning(df, decisions, 'default', input_column)
    reasons = []
    for messages in tqdm(reason_messages, total=len(reason_messages)):
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            stream=False,
            temperature=0.1,
            max_tokens=2048,
        )
        reasons.append(response.choices[0].message.content)
        time.sleep(0.1)

    reasons_df = pd.DataFrame()
    reasons_df['date'] = df['decision_date']
    reasons_df['gold'] = df['reasoning']
    reasons_df['predictions'] = reasons

    if not os.path.exists(f"outputs/deepseek_reasons_{input_column}.xlsx"):
        reasons_df.to_excel(f"outputs/deepseek_reasons_{input_column}.xlsx", sheet_name=model, index=False)
    else:
        with pd.ExcelWriter(f'outputs/deepseek_reasons_{input_column}.xlsx', mode='a', engine='openpyxl',
                            if_sheet_exists='replace') as writer:
            reasons_df.to_excel(writer, sheet_name=model, index=False)


if __name__ == '__main__':
    run('judgment')
