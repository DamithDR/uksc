import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline, AutoConfig

from util.eval import eval_decisions


def run(args):
    df = pd.read_excel('data/UKSC_dataset.xlsx', sheet_name='data')

    responses = []
    pipe = pipeline(
        "text-generation",
        model=args.model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    # pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    # pipe.tokenizer.padding_side = 'left'
    for title in tqdm(zip(df['title']), total=len(df)):
        messages = [
            {"role": "system",
             "content": "You will be given titles of possible court cases happened in UK supreme court. Your task is to check if you have seen the provided case before in your training data and choose either yes or no."
                        "Do not give any other explanation, just say yes or no depending on whether you have seen the particular case before."
             },
            {"role": "user", "content": title},
        ]
        outputs = pipe(
            messages,
            max_new_tokens=10,
            temperature=0.9,
            top_k=20,
            top_p=0.8,
            pad_token_id=pipe.model.config.eos_token_id,
            num_return_sequences=1,
        )
        resp = outputs[0]["generated_text"][-1]['content'].strip()

        responses.append(resp)
        print(resp)

    # df['outputs'] = output_list
    model_name = str(args.model_name).replace('/', '_')
    # df.to_csv(f'outputs/output_{model_name}.tsv', sep='\t', index=False)

    responses_df = pd.DataFrame()
    responses_df['title'] = df['title']
    responses_df['predictions'] = responses

    responses_df.to_excel(f'outputs/{model_name}_decisions.xlsx', sheet_name=model_name, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''judgement prediction in UKSC cases''')
    parser.add_argument('--model_name', required=True, help='model_name')
    args = parser.parse_args()
    run(args)
