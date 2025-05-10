import os
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from retrieval.BM25Retriever import BM25Retriever
from retrieval.RandomRetriever import RandomRetriever
from util.eval import eval_decisions


def get_rag_messages_for_labels(df, retriever, input_column='background', retrieve_mode='random'):
    label_classification_messages = []
    for background, decision, reason, title, legal_area in tqdm(
            zip(df[input_column], df['decision'], df['reasoning'], df['title'], df['legal_area']),
            total=len(df),
            desc="generating label outputs"):
        if retrieve_mode == 'random':
            rag_case = retriever.retrieve()
        elif retrieve_mode == 'bm25':
            rag_case = retriever.retrieve(background, top_k=1)

        mode_string = ''
        messages = [
            {"role": "system",
             "content": "Assume you are a judge at the supreme court in United Kingdom. "
                        "You will be provided UK supreme court appeal cases by the users and your duty is to understand the case background and output your decision label. You will be given an example background of a case and the output of the case for your reference."
                        "Classify whether the provided appeal is allowed or dismissed, select one from following : [allow,dismiss]"
             },
            {"role": "user",
             "content": f"The case title is {title}. Please recognise the appellant and respondents separately using the given title as they have indicated within brackets. {mode_string} Following is the case background, please respond allow/dismiss, do not respond any explanation, other than allow/dismiss. "
                        f"Appeal: {background}"},
            {"role": "user",
             "content": f"Example case background {rag_case[0]['text']}."
                        f"The output of the case: {rag_case[0]['decision_label']}"},
        ]
        label_classification_messages.append(messages)
    return label_classification_messages


def get_messages_for_reasoning(df, decision_labels, retriever, input_column='background', retrieve_mode='random'):
    reasoning_messages = []
    for background, decision, reason, title, legal_area, label in tqdm(
            zip(df[input_column], df['decision'], df['reasoning'], df['title'], df['legal_area'], decision_labels),
            total=len(df),
            desc="generating label outputs"):
        if retrieve_mode == 'bm25':
            rag_case = retriever.retrieve(background, top_k=1)
        else:
            rag_case = retriever.retrieve()
        mode_string = ''
        messages = [
            {"role": "system",
             "content": "Assume you are a judge at the supreme court in United Kingdom. "
                        "You will be provided UK supreme court appeal cases by the users and your duty is to understand the case background and output your decision label. You will be given an example background of a case and the output of the case for your reference."
                        "Classify whether the provided appeal is allowed or dismissed, select one from following : [allow,dismiss]"
             },
            {"role": "user",
             "content": f"The case title is {title}. Please recognise the appellant and respondent seperately using the given title as they have indicated within brackets. {mode_string} Following is the case background, please respond allow/dismiss, do not respond any explanation, other than allow/dismiss. "
                        f"Appeal: {background}"},
            {"role": "assistant", "content": label},
            {"role": "user",
             "content": f"Now generate the reason behind your decision. {mode_string} Do not need to mention your decision label again. Carefully consider the case background and your decided label and only output the reasoning behind your decision."},
            {"role": "user",
             "content": f"Example case background {rag_case[0]['text']}."
                        f"The output of the case: {rag_case[0]['decision_label']}"
                        f"The reasoning of the case: {rag_case[0]['reasoning']}"},
        ]
        reasoning_messages.append(messages)

    return reasoning_messages


def run(model, input_column, retrieve_mode='random'):
    if retrieve_mode == 'bm25':
        retriever = BM25Retriever(
            excel_file="data/historic/historic_data_with_reason.xlsx",
            sheet_name="data",
            text_column="background",
            label_column="decision_label"
        )
    else:
        retriever = RandomRetriever(
            excel_file="data/historic/historic_data_with_reason.xlsx",
            sheet_name="data",
            text_column="background",
            label_column="decision_label"
        )

    client = OpenAI()

    df = pd.read_excel('data/test_data_extended.xlsx', sheet_name='data')
    decision_messages = get_rag_messages_for_labels(df, retriever, input_column)
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

    if not os.path.exists(f"outputs/rag_random_chatgpt_decisions_{input_column}.xlsx"):
        decisions_df.to_excel(f"outputs/rag_random_chatgpt_decisions_{input_column}.xlsx", sheet_name=model, index=False)
    else:
        with pd.ExcelWriter(f'outputs/rag_random_chatgpt_decisions_{input_column}.xlsx', mode='a', engine='openpyxl',
                            if_sheet_exists='replace') as writer:
            decisions_df.to_excel(writer, sheet_name=model, index=False)

    w_recall, w_precision, w_f1, m_f1 = eval_decisions(decisions_df, 'predictions', 'gold')
    with open(f'rag_random_decision_stats.tsv', 'a') as f:
        f.write(
            f'{model}\t{w_recall}\t{w_precision}\t{w_f1}\t{m_f1}\n')

    # reasoning
    reason_messages = get_messages_for_reasoning(df, decisions, retriever, input_column)
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

    if not os.path.exists(f"outputs/rag_random_chatgpt_reasons_{input_column}.xlsx"):
        reasons_df.to_excel(f"outputs/rag_random_chatgpt_reasons_{input_column}.xlsx", sheet_name=model, index=False)
    else:
        with pd.ExcelWriter(f'outputs/rag_random_chatgpt_reasons_{input_column}.xlsx', mode='a', engine='openpyxl',
                            if_sheet_exists='replace') as writer:
            reasons_df.to_excel(writer, sheet_name=model, index=False)


if __name__ == '__main__':

    for model in ['gpt-4-turbo-2024-04-09']:
    # for model in ['gpt-3.5-turbo-0125', "gpt-4-turbo-2024-04-09"]:
        for input_column in ['background','judgment']:
            run(model, input_column,'random')
