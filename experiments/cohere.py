import json
import os
import time

import cohere
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from experiments.open_llms import get_messages_for_labels, get_messages_for_reasoning
from util.eval import eval_decisions


def run(mode='default'):
    COHERE_API_KEY = "<<your-api-key>>"
    client = cohere.ClientV2(COHERE_API_KEY)
    # client = OpenAI()

    model = "command-a-03-2025"
    # model = "gpt-4-turbo-2024-04-09"
    df = pd.read_excel('data/test_data_extended.xlsx', sheet_name='data')
    decision_messages = get_messages_for_labels(df, mode,'judgment')
    decisions = []
    for messages in tqdm(decision_messages, total=len(decision_messages)):
        # response = client.chat.completions.create(
        #     messages=messages,
        #     model=model,
        #     temperature=0.1,
        #     max_tokens=2048,
        # )

        response = client.chat(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )
        content_items = response.message.content
        text_parts = [c.text for c in content_items if c.type == "text"]
        full_text = " ".join(text_parts).strip()
        decisions.append(full_text)
        time.sleep(0.1)

    decisions_df = pd.DataFrame()
    decisions_df['date'] = df['decision_date']
    decisions_df['gold'] = df['decision_label']
    decisions_df['predictions'] = decisions

    if not os.path.exists(f"outputs/cohere_decisions_{mode}.xlsx"):
        decisions_df.to_excel(f"outputs/cohere_decisions_{mode}.xlsx", sheet_name=model, index=False)
    else:
        with pd.ExcelWriter(f'outputs/cohere_decisions_{mode}.xlsx', mode='a', engine='openpyxl',
                            if_sheet_exists='replace') as writer:
            decisions_df.to_excel(writer, sheet_name=model, index=False)

    w_recall, w_precision, w_f1, m_f1 = eval_decisions(decisions_df, 'predictions', 'gold')
    with open(f'decision_stats_{mode}.tsv', 'a') as f:
        f.write(
            f'{model}\t{w_recall}\t{w_precision}\t{w_f1}\t{m_f1}\n')

    # reasoning
    reason_messages = get_messages_for_reasoning(df, decisions, mode,'judgment')
    reasons = []
    for messages in tqdm(reason_messages, total=len(reason_messages)):
        response = client.chat(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )
        content_items = response.message.content
        text_parts = [c.text for c in content_items if c.type == "text"]
        full_text = " ".join(text_parts).strip()
        # outputs.append(full_text)
        reasons.append(full_text)
        time.sleep(0.1)

    reasons_df = pd.DataFrame()
    reasons_df['date'] = df['decision_date']
    reasons_df['gold'] = df['reasoning']
    reasons_df['predictions'] = reasons

    if not os.path.exists(f"outputs/cohere_reasons_{mode}.xlsx"):
        reasons_df.to_excel(f"outputs/cohere_reasons_{mode}.xlsx", sheet_name=model, index=False)
    else:
        with pd.ExcelWriter(f'outputs/cohere_reasons_{mode}.xlsx', mode='a', engine='openpyxl',
                            if_sheet_exists='replace') as writer:
            reasons_df.to_excel(writer, sheet_name=model, index=False)


if __name__ == '__main__':
    mode = 'default'
    run(mode)
