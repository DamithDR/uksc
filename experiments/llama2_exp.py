import argparse

import pandas as pd
import torch
from transformers import pipeline

from util.eval import eval_decisions


def run(args):
    df = pd.read_excel('data/UKSC_dataset.xlsx', sheet_name='data')

    decisions = []
    reasons = []
    pipe = pipeline(
        "text-generation",
        model=args.model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",

    )
    for background, decision, reason in zip(df['background'], df['decision'], df['reasoning']):
        messages = [
            {"role": "system",
             "content": "Assume you are a judge in supreme court in United Kingdom, "
                        "your duty is to understand the following case background and output your "
                        "decision and the reasoning behind it."
                        "In your response, first show if the appeal is allowed or Dismissed. Then provide the legal reasoning behind your judgement"
                        "Please provide your answer in the following format: "
                        "{allow/dismiss}###{Reason}"
             },
            {"role": "user", "content": background},
        ]
        outputs = pipe(
            messages,
            max_new_tokens=2048,
            temperature=0.1,
            pad_token_id=pipe.model.config.eos_token_id,
        )
        resp = outputs[0]["generated_text"][-1]['content'].strip()
        print(f'Response : {resp}\n======================================================')
        split = resp.split('###')
        decisions.append(split[0].strip().lower())
        reasons.append(split[1].strip())
        print(resp)

    # df['outputs'] = output_list
    model_name = str(args.model_name).replace('/', '_')
    # df.to_csv(f'outputs/output_{model_name}.tsv', sep='\t', index=False)

    decisions_df = pd.DataFrame()
    decisions_df['gold'] = df['decision_label']
    decisions_df['predictions'] = decisions

    reasons_df = pd.DataFrame()
    reasons_df['gold'] = df['reasoning']
    reasons_df['predictions'] = reasons

    decisions_df.to_excel(f'outputs/{model_name}_decisions.xlsx', index=False)
    reasons_df.to_excel(f'outputs/{model_name}_reasons.xlsx', index=False)

    w_recall, w_precision, w_f1, m_f1 = eval_decisions(decisions_df, 'predictions', 'gold')

    with open(f'decision_stats.tsv', 'a') as f:
        f.write(
            f'{model_name}\t{w_recall}\t{w_precision}\t{w_f1}\t{m_f1}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''judgement prediction in UKSC cases''')
    parser.add_argument('--model_name', required=True, help='model_name')
    args = parser.parse_args()
    run(args)
