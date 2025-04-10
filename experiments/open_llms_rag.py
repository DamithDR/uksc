import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from rank_bm25 import BM25Okapi
from util.eval import eval_decisions


# Function to load and preprocess the retrieval corpus from a file
def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n\n')  # Split by paragraphs
    corpus = []
    tokenized_corpus = []
    for line in lines:
        if line.strip():
            # Assume each entry in the file is formatted as "Title: ... | Text: ... | Decision: ... | Reasoning: ..."
            parts = line.split('|')
            if len(parts) >= 3:  # Ensure at least title, text, and decision are present
                title = parts[0].replace("Title:", "").strip()
                text = parts[1].replace("Text:", "").strip()
                decision = parts[2].replace("Decision:", "").strip()
                reasoning = parts[3].replace("Reasoning:", "").strip() if len(parts) > 3 else "No reasoning provided."
                corpus.append({"title": title, "text": text, "decision": decision, "reasoning": reasoning})
                tokenized_corpus.append(text.split())
    return corpus, BM25Okapi(tokenized_corpus)


# Retrieve top-k relevant examples using BM25
def retrieve_examples(query, bm25, corpus, k=3):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = scores.argsort()[-k:][::-1]
    return [corpus[i] for i in top_k_indices]


# Format retrieved examples as few-shot prompts
def format_few_shot_examples(examples, include_reasoning=False):
    few_shot_prompt = "\n\nHere are some example cases to guide your decision:\n"
    for ex in examples:
        few_shot_prompt += f"Case Title: {ex['title']}\nAppeal: {ex['text']}\n"
        if include_reasoning:
            few_shot_prompt += f"Reasoning: {ex['reasoning']}\n"
        else:
            few_shot_prompt += f"Decision: {ex['decision']}\n"
        few_shot_prompt += "---\n"
    return few_shot_prompt


# Truncate text to fit within max token length
def truncate_prompt(prompt, tokenizer, max_length=2048):
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) > max_length:
        truncated_tokens = tokens[:max_length - 100]  # Leave room for special tokens and response
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return prompt


def get_messages_for_labels(df, run_mode=None, bm25=None, corpus=None, tokenizer=None):
    label_classification_messages = []
    max_length = 2048  # Adjust based on model's max context length if needed
    for judgment_text, decision, reason, title, legal_area in tqdm(
            zip(df['judgment_text'], df['decision'], df['reasoning'], df['title'], df['legal_area']),
            total=len(df),
            desc="generating label outputs"):
        mode_string = ''
        if run_mode == 'tag':
            mode_string = 'Please recall your law knowledge related to UK legislation and UK case law regarding legal areas : '
            mode_string += str(legal_area).replace(',', ' and ')
            mode_string += '.'

        # Retrieve few-shot examples using BM25 (decisions only)
        few_shot_examples = ""
        if bm25 and corpus:
            retrieved_examples = retrieve_examples(judgment_text, bm25, corpus)
            few_shot_examples = format_few_shot_examples(retrieved_examples, include_reasoning=False)

        # Construct prompt with few-shot examples at the end
        prompt = (
            f"The case title is {title}. Please recognise the appellant and respondents separately using the given title as they have indicated within brackets. {mode_string} "
            f"Following is the case background, please respond allow/dismiss, do not respond with any explanation, only allow/dismiss.\n"
            f"Appeal: {judgment_text}\n{few_shot_examples}"
        )

        # Truncate if necessary
        truncated_prompt = truncate_prompt(prompt, tokenizer, max_length)

        messages = [
            {"role": "system",
             "content": "Assume you are a judge at the supreme court in United Kingdom. "
                        "You will be provided UK supreme court appeal cases by the users and your duty is to understand the case background and output your decision label. "
                        "Classify whether the provided appeal is allowed or dismissed, select one from following: [allow,dismiss]. "
                        "Use the provided examples to guide your decision."
             },
            {"role": "user", "content": truncated_prompt},
        ]
        label_classification_messages.append(messages)
    return label_classification_messages


def get_messages_for_reasoning(df, decision_labels, run_mode=None, bm25=None, corpus=None, tokenizer=None):
    reasoning_messages = []
    max_length = 2048  # Adjust based on model's max context length if needed
    for judgment_text, decision, reason, title, legal_area, label in tqdm(
            zip(df['judgment_text'], df['decision'], df['reasoning'], df['title'], df['legal_area'], decision_labels),
            total=len(df),
            desc="generating reasoning outputs"):
        mode_string = ''
        if run_mode == 'tag':
            mode_string = 'Please recall your law knowledge related to UK legislation and UK case law regarding legal areas : '
            mode_string += str(legal_area).replace(',', ' and ')
            mode_string += '.'

        # Retrieve few-shot examples using BM25
        label_few_shot_examples = ""
        reasoning_few_shot_examples = ""
        if bm25 and corpus:
            retrieved_examples = retrieve_examples(judgment_text, bm25, corpus)
            label_few_shot_examples = format_few_shot_examples(retrieved_examples, include_reasoning=False)
            reasoning_few_shot_examples = format_few_shot_examples(retrieved_examples, include_reasoning=True)

        # Construct label prompt with few-shot examples (decisions only) at the end
        label_prompt = (
            f"The case title is {title}. Please recognise the appellant and respondent separately using the given title as they have indicated within brackets. {mode_string} "
            f"Following is the case background, please respond allow/dismiss, do not respond with any explanation, only allow/dismiss.\n"
            f"Appeal: {judgment_text}\n{label_few_shot_examples}"
        )
        truncated_label_prompt = truncate_prompt(label_prompt, tokenizer, max_length)

        # Construct reasoning prompt with few-shot examples (reasoning only) at the end
        reasoning_prompt = (
            f"Now generate the reason behind your decision. {mode_string} Do not need to mention your decision label again. "
            f"Carefully consider the case background and the provided examples to guide your reasoning, then output only the reasoning behind your decision.\n"
            f"Appeal: {judgment_text}\n{reasoning_few_shot_examples}"
        )
        truncated_reasoning_prompt = truncate_prompt(reasoning_prompt, tokenizer, max_length)

        messages = [
            {"role": "system",
             "content": "Assume you are a judge at the supreme court in United Kingdom. "
                        "You will be provided UK supreme court appeal cases by the users and your duty is to understand the case background and output your decision label. "
                        "Classify whether the provided appeal is allowed or dismissed, select one from following: [allow,dismiss]. "
                        "Use the provided examples to guide your decision and reasoning."
             },
            {"role": "user", "content": truncated_label_prompt},
            {"role": "assistant", "content": label},
            {"role": "user", "content": truncated_reasoning_prompt}
        ]
        reasoning_messages.append(messages)
    return reasoning_messages


def get_chat_template(args):
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cuda_devices
    df = pd.read_excel('data/test_data.xlsx', sheet_name='data')

    # Load retrieval corpus and initialize BM25
    corpus, bm25 = load_corpus(args.retrieval_file) if args.retrieval_file else (None, None)

    tokenizer_mt = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    chat_template = get_chat_template(args)
    if chat_template:
        tokenizer_mt.chat_template = chat_template
    decision_labels = []

    pipe = pipeline(
        "text-generation",
        model=args.model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        tokenizer=tokenizer_mt,
        trust_remote_code=True
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side = 'left'

    label_classification_messages = get_messages_for_labels(df, args.run_mode, bm25, corpus, tokenizer_mt)

    print(f'{args.model_name} : Generating decision labels')
    decision_outputs = pipe(
        label_classification_messages,
        max_new_tokens=2048,
        temperature=0.1,
        pad_token_id=pipe.model.config.eos_token_id,
        num_return_sequences=1,
        do_sample=True,
        batch_size=args.batch_size,
        truncation=True,
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

    w_recall, w_precision, w_f1, m_f1 = eval_decisions(decisions_df, 'predictions', 'gold')
    with open(f'decision_stats_{args.run_mode}.tsv', 'a') as f:
        f.write(
            f'{model_name}\t{round(w_recall, 2)}\t{round(w_precision, 2)}\t{round(w_f1, 2)}\t{round(m_f1, 2)}\n')

    reasoning_messages = get_messages_for_reasoning(df, decision_labels, args.run_mode, bm25, corpus, tokenizer_mt)
    print(f'{args.model_name} : Generating Reasons')
    reasoning_outputs = pipe(
        reasoning_messages,
        max_new_tokens=2048,
        temperature=0.1,
        pad_token_id=pipe.model.config.eos_token_id,
        num_return_sequences=1,
        do_sample=True,
        batch_size=int(args.batch_size / 2),
        truncation=True,
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
    parser = argparse.ArgumentParser(description='Judgement prediction in UKSC cases with few-shot RAG')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--visible_cuda_devices', type=str, default="0,1,2", required=False,
                        help='Visible CUDA devices')
    parser.add_argument('--batch_size', type=int, required=False, default=2, help='Batch size')
    parser.add_argument('--run_mode', type=str, required=False, default=None, help='Mode of prompt')
    parser.add_argument('--retrieval_file', type=str, required=False, default=None,
                        help='Path to file for BM25 retrieval')
    args = parser.parse_args()
    run(args)