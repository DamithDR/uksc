import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline, AutoConfig, AutoTokenizer

from util.eval import eval_decisions


def run(args):
    df = pd.read_excel('data/UKSC_dataset.xlsx', sheet_name='data')

    tokenizer_mt = AutoTokenizer.from_pretrained(args.model_name)
    chat_template = None
    if str(args.model_name).__contains__('mistral'):
        chat_template = open('templates/mistral-instruct.jinja').read()
    elif str(args.model_name).__contains__('falcon'):
        chat_template = open('templates/falcon-instruct.jinja').read()
    elif str(args.model_name).__contains__('Llama-2'):
        chat_template = open('templates/llama-2-chat.jinja').read()
    if chat_template:
        tokenizer_mt.chat_template = chat_template
    decisions = []
    reasons = []
    pipe = pipeline(
        "text-generation",
        model=args.model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        tokenizer=tokenizer_mt
    )
    # pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    # pipe.tokenizer.padding_side = 'left'
    for background, decision, reason in tqdm(zip(df['background'], df['decision'], df['reasoning']), total=len(df)):
        messages = [
            {"role": "system",
             "content": "Assume you are a judge at the supreme court in United Kingdom. "
                        "You will be provided UK supreme court appeal cases by the users and your duty is to understand the case background and output your decision and the reasoning behind it."
                        "First classify whether the case is allowed or dismissed, select one from following : [allowed,dismissed]"
                        "Output the case classification label followed by the delimiter '###'. After the delimiter, provide a legal explanation for your classification decision."
                        "Example output: [Your classification label]###[Your explanation]"
             },
            {"role": "user", "content": background},
        ]
        outputs = pipe(
            messages,
            max_new_tokens=2048,
            temperature=0.9,
            top_k=20,
            top_p=0.8,
            pad_token_id=pipe.model.config.eos_token_id,
            num_return_sequences=1,
        )
        resp = outputs[0]["generated_text"][-1]['content'].strip()
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
    # parser.add_argument('--batch_size', type=int, required=False, default=8, help='batch_size')
    args = parser.parse_args()
    run(args)
