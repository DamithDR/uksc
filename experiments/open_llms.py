import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer

from util.eval import eval_decisions


def get_messages_for_labels(df, run_mode=None):
    label_classification_messages = []
    for background, decision, reason, title, legal_area in tqdm(
            zip(df['background'], df['decision'], df['reasoning'], df['title'], df['legal_area']),
            total=len(df),
            desc="generating label outputs"):
        mode_string = ''
        if run_mode == 'tag':
            mode_string = 'Please recall your law knowledge related to UK legislation and UK case law regarding legal areas : '
            mode_string += str(legal_area).replace(',', ' and ')
            mode_string += '.'
        messages = [
            {"role": "system",
             "content": "Assume you are a judge at the supreme court in United Kingdom. "
                        "You will be provided UK supreme court appeal cases by the users and your duty is to understand the case background and output your decision label."
                        "Classify whether the provided appeal is allowed or dismissed, select one from following : [allow,dismiss]"
             },
            {"role": "user",
             "content": f"The case title is {title}. Please recognise the appellant and respondents seperately using the given title as they have indicated within brackets. {mode_string} Following is the case background, please respond allow/dismiss, do not respond any explanation, other than allow/dismiss. "
                        f"Appeal: {background}"},
        ]
        label_classification_messages.append(messages)
    return label_classification_messages


def get_messages_for_reasoning(df, decision_labels, run_mode=None):
    reasoning_messages = []
    for background, decision, reason, title, legal_area, label in tqdm(
            zip(df['background'], df['decision'], df['reasoning'], df['title'], df['legal_area'], decision_labels),
            total=len(df),
            desc="generating label outputs"):
        mode_string = ''
        if run_mode == 'tag':
            mode_string = 'Please recall your law knowledge related to UK legislation and UK case law regarding legal areas : '
            mode_string += str(legal_area).replace(',', ' and ')
            mode_string += '.'
        messages = [
            {"role": "system",
             "content": "Assume you are a judge at the supreme court in United Kingdom. "
                        "You will be provided UK supreme court appeal cases by the users and your duty is to understand the case background and output your decision label."
                        "Classify whether the provided appeal is allowed or dismissed, select one from following : [allow,dismiss]"
             },
            {"role": "user",
             "content": f"The case title is {title}. Please recognise the appellant and respondent seperately using the given title as they have indicated within brackets. {mode_string} Following is the case background, please respond allow/dismiss, do not respond any explanation, other than allow/dismiss. "
                        f"Appeal: {background}"},
            {"role": "assistant", "content": label},
            {"role": "user",
             "content": f"Now generate the reason behind your decision. {mode_string} Do not need to mention your decision label again. Carefully consider the case background and your decided label and only output the reasoning behind your decision."}
        ]
        reasoning_messages.append(messages)
    return reasoning_messages


def get_chat_template():
    # https://github.com/chujiezheng/chat_templates/tree/main/chat_templates
    chat_template = None
    if str(args.model_name).__contains__('mistral'):
        chat_template = open('templates/mistral-instruct.jinja').read()
    elif str(args.model_name).__contains__('falcon'):
        chat_template = open('templates/falcon-instruct.jinja').read()
    elif str(args.model_name).__contains__('Llama-2') or str(args.model_name).__contains__('Saul-7B'):
        chat_template = open('templates/llama-2-chat.jinja').read()
    elif str(args.model_name).__contains__('Meta-Llama-3'):
        chat_template = open('templates/llama-3-instruct.jinja').read()
    elif str(args.model_name).__contains__('Phi-3'):
        chat_template = open('templates/phi-3.jinja').read()
    return chat_template


def run(args):
    print(f'{args.model_name} : Running Started | Run mode : {args.run_mode}')
    model_name = str(args.model_name).split('/')[1] if str(args.model_name).__contains__('/') else str(args.model_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cuda_devices  # set the devices you need to run
    df = pd.read_excel('data/test_data.xlsx', sheet_name='data')

    tokenizer_mt = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    chat_template = get_chat_template()
    if chat_template:
        tokenizer_mt.chat_template = chat_template
    decision_labels = []
    # https://github.com/Dao-AILab/flash-attention/issues/246 - use this : pip install flash_attn --no-build-isolation
    # https://github.com/microsoft/Phi-3CookBook/issues/115 - phi-3 flash attention issue
    pipe = pipeline(
        "text-generation",
        model=args.model_name,
        model_kwargs={"torch_dtype": torch.bfloat16,
                      # "attn_implementation": "flash_attention_2" if str(args.model_name).__contains__(
                      #     'Phi-3') else None
                      },
        device_map="auto",
        tokenizer=tokenizer_mt,
        trust_remote_code=True
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side = 'left'

    label_classification_messages = get_messages_for_labels(df, args.run_mode)

    print(f'{args.model_name} : Generating decision labels')
    decision_outputs = pipe(
        label_classification_messages,
        max_new_tokens=2048,
        temperature=0.1,
        pad_token_id=pipe.model.config.eos_token_id,
        num_return_sequences=1,
        do_sample=True,
        batch_size=args.batch_size
    )
    for output in tqdm(decision_outputs, total=len(decision_outputs), desc="extracting label outputs"):
        resp = output[0]["generated_text"][-1]['content'].lower().strip()
        decision_labels.append(resp)

    decisions_df = pd.DataFrame()
    decisions_df['date'] = df['decision_date']
    decisions_df['gold'] = df['decision_label']
    decisions_df['predictions'] = decision_labels

    if not os.path.exists(f"outputs/decisions_{args.run_mode}.xlsx"):
        decisions_df.to_excel(f"outputs/decisions_{args.run_mode}.xlsx", sheet_name=f"{model_name}", index=False)
    else:
        with pd.ExcelWriter(f'outputs/decisions_{args.run_mode}.xlsx', mode='a', engine='openpyxl',
                            if_sheet_exists='replace') as writer:
            decisions_df.to_excel(writer, sheet_name=f"{model_name}", index=False)

    # save results of label evaluation
    w_recall, w_precision, w_f1, m_f1 = eval_decisions(decisions_df, 'predictions', 'gold')
    with open(f'decision_stats_{args.run_mode}.tsv', 'a') as f:
        f.write(
            f'{model_name}\t{round(w_recall, 2)}\t{round(w_precision, 2)}\t{round(w_f1, 2)}\t{round(m_f1, 2)}\n')

    reasoning_messages = get_messages_for_reasoning(df, decision_labels, args.run_mode)
    print(f'{args.model_name} : Generating Reasons')
    reasoning_outputs = pipe(
        reasoning_messages,
        max_new_tokens=2048,
        temperature=0.1,
        pad_token_id=pipe.model.config.eos_token_id,
        num_return_sequences=1,
        do_sample=True,
        batch_size=int(args.batch_size / 2)
    )

    reasons = []
    for output in tqdm(reasoning_outputs, total=len(reasoning_outputs), desc="extracting reasoning outputs"):
        resp = output[0]["generated_text"][-1]['content'].strip()
        reasons.append(resp)

    reasons_df = pd.DataFrame()
    reasons_df['date'] = df['decision_date']
    reasons_df['gold'] = df['reasoning']
    reasons_df['predictions'] = reasons

    if not os.path.exists(f"outputs/reasons_{args.run_mode}.xlsx"):
        reasons_df.to_excel(f"outputs/reasons_{args.run_mode}.xlsx", sheet_name=f"{model_name}", index=False)
    else:
        with pd.ExcelWriter(f'outputs/reasons_{args.run_mode}.xlsx', mode='a', engine='openpyxl',
                            if_sheet_exists='replace') as writer:
            reasons_df.to_excel(writer, sheet_name=f"{model_name}", index=False)
    print(f'{args.model_name} : Outputs saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''judgement prediction in UKSC cases''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--visible_cuda_devices', type=str, default="0,1,2", required=False, help='model_name')
    parser.add_argument('--batch_size', type=int, required=False, default=8, help='batch_size')
    parser.add_argument('--run_mode', type=str, required=False, default=None, help='mode of prompt')
    args = parser.parse_args()
    run(args)
