import pandas as pd
from pandas import DataFrame
from transformers import LEDTokenizer, LEDForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import torch
import os


INPUT_COLUMN = "background"
TRAIN_PATH = "data/historic/historic_data_with_reason.xlsx"
TEST_PATH = "data/test_data_extended.xlsx"
SHEET_NAME = "data"
MODEL_NAME = "allenai/led-base-16384"
MAX_INPUT_LENGTH = 8192
MAX_TARGET_LENGTH = 2048
MAX_OUTPUT_LENGTH  = 2048
OUTPUT_DIR = "./led_reasoning_model"

#  Load data
train_df = pd.read_excel(TRAIN_PATH, sheet_name=SHEET_NAME)
inference_df = pd.read_excel(TEST_PATH, sheet_name=SHEET_NAME)

#  Prompt formatting functions
def create_prompt(row, column):
    return f"Decision of the following case is {row['decision_label']}. Then explain the reasoning for the case: {row[column]}"

def create_target(row):
    return row["reasoning"]

#  Prepare dataset for Hugging Face Datasets
train_df = train_df.dropna(subset=["reasoning", "decision_label", INPUT_COLUMN])
train_df["input_text"] = train_df.apply(lambda row: create_prompt(row, INPUT_COLUMN), axis=1)
train_df["target_text"] = train_df.apply(create_target, axis=1)

# Convert to HF Dataset
train_dataset = Dataset.from_pandas(train_df[["input_text", "target_text"]])

#  Tokenizer and Model
tokenizer = LEDTokenizer.from_pretrained(MODEL_NAME)
model = LEDForConditionalGeneration.from_pretrained(MODEL_NAME)

def tokenize(batch):
    inputs = tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    )
    targets = tokenizer(
        batch["target_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_TARGET_LENGTH,
        return_tensors="pt"
    )

    inputs["labels"] = targets["input_ids"]
    inputs["global_attention_mask"] = torch.zeros_like(inputs["input_ids"])
    inputs["global_attention_mask"][:, 0] = 1  # global attention on first token

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": inputs["labels"],
        "global_attention_mask": inputs["global_attention_mask"],
    }

#  Tokenize dataset
train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["input_text", "target_text"])

#  Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,  # Reduce if memory is limited
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

#  Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

#  Train
trainer.train()

#  Save model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


#  Load data
df = pd.read_excel(TEST_PATH, sheet_name=SHEET_NAME).dropna(subset=[INPUT_COLUMN, "decision_label"])


MODEL_PATH = "./led_reasoning_model"
#  Load tokenizer and model
tokenizer = LEDTokenizer.from_pretrained(MODEL_PATH)
model = LEDForConditionalGeneration.from_pretrained(MODEL_PATH)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Generate reasoning
results = []
for _, row in df.iterrows():
    prompt = f"Decision of the following case is {row['decision_label']}. Then explain the reasoning for the case: {row[INPUT_COLUMN]}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=4,
            early_stopping=True
        )
    reasoning = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    results.append(reasoning)


results_df = DataFrame()
results_df['gold'] = df[INPUT_COLUMN]
results_df["predictions"] = results
results_df.to_excel("led_generated_reasoning_results.xlsx", index=False)
print("Inference complete. Output saved to generated_reasoning_results.xlsx")

