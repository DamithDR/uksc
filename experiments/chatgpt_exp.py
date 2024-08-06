import json
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


client = OpenAI()

responses = []
model = 'gpt-3.5-turbo-0125'
# model="gpt-4-0125-preview",
df = pd.read_excel('data/UKSC_dataset.xlsx', sheet_name='data')
for background, decision, reason in tqdm(zip(df['background'], df['decision'], df['reasoning']), total=len(df)):
    response = client.chat.completions.create(
        messages=[
            {"role": "system",
             "content": "Assume you are a judge in supreme court in United Kingdom, your duty is to understand the following case background and output the likely outcome and the reasoning behind it."},
            {"role": "user", "content": background},
        ],
        model=model,
        temperature=0.9,
        max_tokens=2048,
    )

    resp = response.choices[0].message.content
    print(resp)
    responses.append(resp)
    time.sleep(0.2)

df['outputs'] = responses
df.to_csv(f'outputs/chatgpt_outputs.tsv', sep='\t', index=False)
