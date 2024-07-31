import argparse
import pandas as pd

from transformers import AutoTokenizer, pipeline
import torch


def run(args):
    df = pd.read_excel('data/UKSC_dataset.xlsx', sheet_name='data')

    output_list = []
    pipe = pipeline(
        "text-generation",
        model=args.model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",

    )
    for background, decision, reason in zip(df['background'], df['decision'], df['reasoning']):
        messages = [
            {"role": "system",
             "content": "Assume you are a judge in supreme court in United Kingdom, your duty is to understand the following case background and output the likely outcome and the reasoning behind it."},
            {"role": "user", "content": background},
        ]
        outputs = pipe(
            messages,
            max_new_tokens=2048,
            pad_token_id=pipe.model.config.eos_token_id,
        )
        output_list.append(outputs[0]["generated_text"][-1])
    df['outputs'] = output_list
    model_name = str(args.model_name).replace('/', '_')
    df.to_csv(f'outputs/output_{model_name}.tsv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''judgement prediction in UKSC cases''')
    parser.add_argument('--model_name', required=True, help='model_name')
    args = parser.parse_args()
    run(args)
