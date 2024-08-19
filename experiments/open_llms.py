import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer

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
                        "Classify whether the provided appeal is allowed or dismissed, select one from following : [allow,dismiss]."
             },
            {"role": "user",
             "content": f"The case title is {title}. Please understand the appellant and respondents using the title. Following is the case background, please respond allow/dismiss, do not respond any explanation, other than allow/dismiss. appeal: {background}"},
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
                        "Classify whether the provided appeal is allowed or dismissed, select one from following : [allow,dismiss]."
             },
            {"role": "user",
             "content": f"The case title is {title}. Please understand the appellant and respondents using the title. Following is the case background, please respond allow/dismiss, do not respond any explanation, other than allow/dismiss. appeal: {background}"},
            {"role": "assistant", "content": label},
            {"role": "user",
             "content": "Now please generate the reason behind your decision. Carefully consider the case background and your decided label and output the reasoning behind your decision."}
        ]
        reasoning_messages.append(messages)
    return reasoning_messages


def get_chat_template():
    chat_template = None
    if str(args.model_name).__contains__('mistral'):
        chat_template = open('templates/mistral-instruct.jinja').read()
    elif str(args.model_name).__contains__('falcon'):
        chat_template = open('templates/falcon-instruct.jinja').read()
    elif str(args.model_name).__contains__('Llama-2'):
        chat_template = open('templates/llama-2-chat.jinja').read()
    return chat_template


def run(args):
    model_name = str(args.model_name).replace('/', '_')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cuda_devices  # set the devices you need to run
    df = pd.read_excel('data/UKSC_dataset.xlsx', sheet_name='data')

    df = df[:40]

    tokenizer_mt = AutoTokenizer.from_pretrained(args.model_name)
    chat_template = get_chat_template()
    if chat_template:
        tokenizer_mt.chat_template = chat_template

    decision_labels = []
    pipe = pipeline(
        "text-generation",
        model=args.model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        tokenizer=tokenizer_mt
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side = 'left'

    label_classification_messages = get_messages_for_labels(df)
    decision_outputs = pipe(
        label_classification_messages,
        max_new_tokens=2048,
        temperature=0.1,
        pad_token_id=pipe.model.config.eos_token_id,
        num_return_sequences=1,
        batch_size=args.batch_size
    )
    for output in tqdm(decision_outputs, total=len(decision_outputs), desc="extracting label outputs"):
        resp = output[0]["generated_text"][-1]['content'].lower().strip()
        decision_labels.append(resp)

    decisions_df = pd.DataFrame()
    decisions_df['gold'] = df['decision_label']
    decisions_df['predictions'] = decision_labels
    decisions_df.to_excel(f'outputs/decisions.xlsx', sheet_name=f"{model_name}", index=False)

    # save results of label evaluation
    w_recall, w_precision, w_f1, m_f1 = eval_decisions(decisions_df, 'predictions', 'gold')
    with open(f'decision_stats.tsv', 'a') as f:
        f.write(
            f'{model_name}\t{round(w_recall, 2)}\t{round(w_precision, 2)}\t{round(w_f1, 2)}\t{round(m_f1, 2)}\n')

    reasoning_messages = get_messages_for_reasoning(df, decision_labels)
    reasoning_outputs = pipe(
        reasoning_messages,
        max_new_tokens=2048,
        temperature=0.1,
        pad_token_id=pipe.model.config.eos_token_id,
        num_return_sequences=1,
        batch_size=int(args.batch_size / 2)
    )

    reasons = []
    for output in tqdm(reasoning_outputs, total=len(reasoning_outputs), desc="extracting reasoning outputs"):
        resp = output[0]["generated_text"][-1]['content'].strip()
        reasons.append(resp)

    reasons_df = pd.DataFrame()
    reasons_df['gold'] = df['reasoning']
    reasons_df['predictions'] = reasons

    reasons_df.to_excel(f'outputs/reasons.xlsx', sheet_name=f"{model_name}", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''judgement prediction in UKSC cases''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--visible_cuda_devices', type=str, default="0,1,2", required=False, help='model_name')
    parser.add_argument('--batch_size', type=int, required=False, default=8, help='batch_size')
    args = parser.parse_args()
    run(args)
