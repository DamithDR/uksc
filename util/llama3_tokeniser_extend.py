import argparse
import os

from transformers import AutoTokenizer, AutoModelForCausalLM


def run(args):
    model_name = str(args.model_name).split('/')[1] if str(args.model_name).__contains__('/') else str(args.model_name)
    model_path = f'local_models/{model_name}'
    if not os.path.exists(model_path): os.mkdir(model_path)

    tokenizer_mt = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer_mt.add_special_tokens({"pad_token": "<pad>"})

    llm_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    llm_model.config.pad_token_id = tokenizer_mt.pad_token_id

    llm_model.resize_token_embeddings(len(tokenizer_mt))
    tokenizer_mt.save_pretrained(model_path)
    llm_model.save_pretrained(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''change llama3 tokeniser''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    args = parser.parse_args()
    run(args)
