import json
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from util.eval import eval_decisions

def get_messages_for_labels(df):
    label_classification_messages = []
    for background, decision, reason, title in tqdm(zip(df['background'], df['decision'], df['reasoning'], df['title']),
                                                    total=len(df),
                                                    desc="generating label outputs"):
        messages = [
            {"role": "system",
             "content": "Assume you are a judge at the supreme court in United Kingdom. "
                        "You will be provided UK supreme court appeal cases by the users and your duty is to understand the case background and output your decision label."
                        "Classify whether the provided appeal is allowed or dismissed, select one from following : [allow,dismiss]"
             },
            {"role": "user",
             "content": f"The case title is {title}. Please recognise the appellant and respondents seperately using the given title as they have indicated within brackets. Following is the case background, please respond allow/dismiss, do not respond any explanation, other than allow/dismiss. "
                        f"Appeal: {background}"},
        ]
        label_classification_messages.append(messages)
    return label_classification_messages


def get_messages_for_reasoning(df, decision_labels):
    reasoning_messages = []
    for background, decision, reason, title, label in tqdm(
            zip(df['background'], df['decision'], df['reasoning'], df['title'], decision_labels), total=len(df),
            desc="generating label outputs"):
        messages = [
            {"role": "system",
             "content": "Assume you are a judge at the supreme court in United Kingdom. "
                        "You will be provided UK supreme court appeal cases by the users and your duty is to understand the case background and output your decision label."
                        "Classify whether the provided appeal is allowed or dismissed, select one from following : [allow,dismiss]"
             },
            {"role": "user",
             "content": f"The case title is {title}. Please recognise the appellant and respondent seperately using the given title as they have indicated within brackets. Following is the case background, please respond allow/dismiss, do not respond any explanation, other than allow/dismiss. "
                        f"Appeal: {background}"},
            {"role": "assistant", "content": label},
            {"role": "user",
             "content": "Now generate the reason behind your decision. Do not need to mention your decision label again. Carefully consider the case background and your decided label and only output the reasoning behind your decision."}
        ]
        reasoning_messages.append(messages)
    return reasoning_messages


def run():
    client = OpenAI()

    decisions = []
    reasons = []
    model = 'gpt-3.5-turbo-0125'
    # model="gpt-4-0125-preview",
    df = pd.read_excel('data/test_data.xlsx', sheet_name='data')
    for background, decision, reason in tqdm(zip(df['background'], df['decision'], df['reasoning']), total=len(df)):
        response = client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": "Assume you are a judge in supreme court in United Kingdom, "
                            "your duty is to understand the following case background and output the "
                            "outcome and the reasoning behind it."
                            "In your response, first show if the appeal is Approved or Dismissed. Then provide the legal reasoning behind your judgement"
                            "Please provide your answer in the following format: "
                            "{allow/dismiss}###{Reason}"
                 },
                {"role": "user", "content": background},
            ],
            model=model,
            temperature=0.1,
            max_tokens=2048,
        )

        resp = response.choices[0].message.content
        print(resp)
        split = resp.split('###')

        decisions.append(split[0].strip().lower())
        reasons.append(split[1].strip())
        time.sleep(0.2)

    decisions_df = pd.DataFrame()
    decisions_df['gold'] = df['decision_label']
    decisions_df['predictions'] = decisions

    reasons_df = pd.DataFrame()
    reasons_df['gold'] = df['reasoning']
    reasons_df['predictions'] = reasons

    decisions_df.to_excel(f'outputs/chatgpt_decisions.xlsx', index=False)
    reasons_df.to_excel(f'outputs/chatgpt_reasons.xlsx', index=False)

    w_recall, w_precision, w_f1, m_f1 = eval_decisions(decisions_df, 'predictions', 'gold')

    with open(f'decision_stats.tsv', 'a') as f:
        f.write(
            f'{model}\t{w_recall}\t{w_precision}\t{w_f1}\t{m_f1}\n')

if __name__ == '__main__':
    run()
