
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from transformers import AutoTokenizer


def preprocess(example):
    # Tokenize input prompt
    model_input = tokenizer(
        example['input'],
        truncation=True,
        padding="max_length",
        max_length=8192
    )

    # Tokenize output (reasoning)
    with tokenizer.as_target_tokenizer():
        label = tokenizer(
            example['output'],
            truncation=True,
            padding="max_length",
            max_length=2048
        )

    model_input["labels"] = label["input_ids"]
    return model_input


def generate_reasoning(text):
        prompt = f"Explain the legal reasoning: {text}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length",
                           max_length=MAX_INPUT_LENGTH).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=MAX_TARGET_LENGTH,
                num_beams=4
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == '__main__':

    column = 'background'

    MODEL_NAME = "google/long-t5-tglobal-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    MAX_INPUT_LENGTH = 2048
    MAX_TARGET_LENGTH = 512

    train_df = pd.read_excel('data/historic/historic_data_with_reason.xlsx', sheet_name='data')
    train_df['prompt'] = train_df.apply(
        lambda
            row: f"Decision of the following case is {row['decision_label']}. Then explain the reasoning for the case: {row[column]}",
        axis=1
    )

    # Format for HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df[['prompt', 'reasoning']].rename(columns={
        "prompt": "input",
        "reasoning": "output"
    }))

    tokenized_train = train_dataset.map(preprocess, batched=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir="./longt5-legal-finetuned",
        per_device_train_batch_size=1,
        num_train_epochs=10,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=1,
        save_strategy="epoch",
        learning_rate=5e-5,
        fp16=True,  # if using GPU with fp16 support
        evaluation_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer
    )

    trainer.train()

    # Load inference set
    inference_df = pd.read_excel('data/test_data_extended.xlsx', sheet_name='data')


    inference_df['prompt'] = inference_df.apply(
        lambda
            row: f"Decision of the following case is {row['decision_label']}. Then explain the reasoning for the case: {row[column]}",
        axis=1
    )

    tqdm.pandas()
    reasoning_df = pd.DataFrame()
    reasoning_df['gold'] = inference_df['reasoning']
    reasoning_df['predictions'] = inference_df['prompt'].progress_apply(generate_reasoning)


    # Save results
    reasoning_df.to_excel("inference_with_reasoning.xlsx", sheet_name="data", index=False)