import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    DataCollatorForSeq2Seq, pipeline
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel
import torch

# -------------------- Config --------------------
MODEL_NAME = "google/flan-t5-base"  # or flan-t5-xl
INPUT_COLUMN = "judgment"  # or "background"
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 3
OUTPUT_DIR = "./flan_t5_lora"
# ------------------------------------------------

# Load dataset
train_df = pd.read_excel('data/historic/historic_data_with_reason.xlsx', sheet_name='data')
train_df = train_df.dropna(subset=[INPUT_COLUMN, "decision_label", "reasoning"])

# Format prompt-target pairs
def format_example(row):
    prompt = f"Decision of the following case is {row['decision_label']}. Then explain the reasoning for the case: {row[INPUT_COLUMN]}"
    target = row['reasoning']
    return {"prompt": prompt, "target": target}

train_data = train_df.apply(format_example, axis=1).tolist()
train_dataset = Dataset.from_list(train_data)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load base model with 8-bit quantization for LoRA
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map="auto")
model = prepare_model_for_kbit_training(model)

# Configure LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],  # for T5 use ["q", "v"], for other models it may differ
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Tokenization
def preprocess(example):
    inputs = tokenizer(example["prompt"], max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True)
    targets = tokenizer(example["target"], max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = train_dataset.map(preprocess, batched=True)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training args
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=20,
    learning_rate=5e-4,
    fp16=True,
    save_total_limit=2,
    save_strategy="epoch"
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Train
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"LoRA fine-tuning complete. Saved to: {OUTPUT_DIR}")


# # -------------------- Config --------------------
BASE_MODEL = "google/flan-t5-base"
LORA_DIR = "./flan_t5_lora"
INPUT_COLUMN = "background"  # or "background"
EXCEL_PATH = "data/test_data_extended.xlsx"
SHEET_NAME = "data"
OUTPUT_PATH = "lora_flan_t5_inference_jud.xlsx"
# # ------------------------------------------------

# Load test data
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df = df.dropna(subset=[INPUT_COLUMN, "decision_label"])

# Prepare prompts
prompts = [
    f"Decision of the following case is {row['decision_label']}. Then explain the reasoning for the case: {row[INPUT_COLUMN]}"
    for _, row in df.iterrows()
]

# Load LoRA model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, load_in_8bit=True, device_map="auto")
model = PeftModel.from_pretrained(base_model, LORA_DIR)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device_map='auto')

# Run batched inference
results = []
for prompt in prompts:
    output = pipe(prompt, max_length=512, truncation=True)[0]["generated_text"]
    results.append(output)

df["generated_reasoning"] = results
df.to_excel(OUTPUT_PATH,sheet_name='data', index=False)
print(f"Inference complete. Results saved to: {OUTPUT_PATH}")

