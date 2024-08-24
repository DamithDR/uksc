import json
import os
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from experiments.open_llms import get_messages_for_labels, get_messages_for_reasoning
from util.eval import eval_decisions


# def get_messages_for_labels(df):
#     label_classification_messages = []
#     for background, decision, reason, title in tqdm(zip(df['background'], df['decision'], df['reasoning'], df['title']),
#                                                     total=len(df),
#                                                     desc="generating label outputs"):
#         messages = [
#             {"role": "system",
#              "content": "Assume you are a judge at the supreme court in United Kingdom. "
#                         "You will be provided UK supreme court appeal cases by the users and your duty is to understand the case background and output your decision label."
#                         "Classify whether the provided appeal is allowed or dismissed, select one from following : [allow,dismiss]"
#              },
#             {"role": "user",
#              "content": f"The case title is {title}. Please recognise the appellant and respondents seperately using the given title as they have indicated within brackets. Following is the case background, please respond allow/dismiss, do not respond any explanation, other than allow/dismiss. "
#                         f"Appeal: {background}"},
#         ]
#         label_classification_messages.append(messages)
#     return label_classification_messages
#
#
# def get_messages_for_reasoning(df, decision_labels):
#     reasoning_messages = []
#     for background, decision, reason, title, label in tqdm(
#             zip(df['background'], df['decision'], df['reasoning'], df['title'], decision_labels), total=len(df),
#             desc="generating label outputs"):
#         messages = [
#             {"role": "system",
#              "content": "Assume you are a judge at the supreme court in United Kingdom. "
#                         "You will be provided UK supreme court appeal cases by the users and your duty is to understand the case background and output your decision label."
#                         "Classify whether the provided appeal is allowed or dismissed, select one from following : [allow,dismiss]"
#              },
#             {"role": "user",
#              "content": f"The case title is {title}. Please recognise the appellant and respondent seperately using the given title as they have indicated within brackets. Following is the case background, please respond allow/dismiss, do not respond any explanation, other than allow/dismiss. "
#                         f"Appeal: {background}"},
#             {"role": "assistant", "content": label},
#             {"role": "user",
#              "content": "Now generate the reason behind your decision. Do not need to mention your decision label again. Carefully consider the case background and your decided label and only output the reasoning behind your decision."}
#         ]
#         reasoning_messages.append(messages)
#     return reasoning_messages


def run(mode='default'):
    client = OpenAI()

    model = 'gpt-3.5-turbo-0125'
    # model = "gpt-4-turbo-2024-04-09"
    df = pd.read_excel('data/test_data.xlsx', sheet_name='data')
    decision_messages = get_messages_for_labels(df, mode)
    decisions = []
    for messages in tqdm(decision_messages, total=len(decision_messages)):
        response = client.chat.completions.create(
            messages=messages,
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

    if not os.path.exists(f"outputs/chatgpt_decisions_{mode}.xlsx"):
        decisions_df.to_excel(f"outputs/chatgpt_decisions_{mode}.xlsx", sheet_name=model, index=False)
    else:
        with pd.ExcelWriter(f'outputs/chatgpt_decisions_{mode}.xlsx', mode='a', engine='openpyxl',
                            if_sheet_exists='replace') as writer:
            decisions_df.to_excel(writer, sheet_name=model, index=False)

    w_recall, w_precision, w_f1, m_f1 = eval_decisions(decisions_df, 'predictions', 'gold')
    with open(f'decision_stats_{mode}.tsv', 'a') as f:
        f.write(
            f'{model}\t{w_recall}\t{w_precision}\t{w_f1}\t{m_f1}\n')

    # reasoning
    reason_messages = get_messages_for_reasoning(df, decisions, mode)
    reasons = []
    for messages in tqdm(reason_messages, total=len(reason_messages)):
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.1,
            max_tokens=2048,
        )
        reasons.append(response.choices[0].message.content)
        time.sleep(0.1)

    reasons_df = pd.DataFrame()
    reasons_df['date'] = df['decision_date']
    reasons_df['gold'] = df['reasoning']
    reasons_df['predictions'] = reasons

    if not os.path.exists(f"outputs/chatgpt_reasons_{mode}.xlsx"):
        reasons_df.to_excel(f"outputs/chatgpt_reasons_{mode}.xlsx", sheet_name=model, index=False)
    else:
        with pd.ExcelWriter(f'outputs/chatgpt_reasons_{mode}.xlsx', mode='a', engine='openpyxl',
                            if_sheet_exists='replace') as writer:
            reasons_df.to_excel(writer, sheet_name=model, index=False)


if __name__ == '__main__':
    mode = 'tag'
    run(mode)
