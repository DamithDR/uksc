import json
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from util.eval import eval_decisions

client = OpenAI()

decisions = []
reasons = []
model = 'gpt-3.5-turbo-0125'
# model="gpt-4-0125-preview",
df = pd.read_excel('data/UKSC_dataset.xlsx', sheet_name='data')
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


