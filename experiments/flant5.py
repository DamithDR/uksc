import pandas as pd
from pandas import DataFrame
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Configuration
MODEL_NAME = "google/flan-ul2"  # or flan-t5-xl, flan-ul2, etc.
INPUT_COLUMN = "judgment"  # or "background"
EXCEL_PATH = "data/test_data_extended.xlsx"
SHEET_NAME = "data"
MAX_INPUT_LENGTH = 2048
MAX_OUTPUT_LENGTH = 512
BATCH_SIZE = 4  # adjust based on available memory

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load test data
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df = df.dropna(subset=[INPUT_COLUMN, "decision_label"])

# Format prompts
prompts = [
    f"Decision of the following case is {row['decision_label']}. Then explain the reasoning for the case: {row[INPUT_COLUMN]}"
    for _, row in df.iterrows()
]

# Run inference in batches
results = []
for i in range(0, len(prompts), BATCH_SIZE):
    batch_prompts = prompts[i:i + BATCH_SIZE]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=4,
            early_stopping=True
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results.extend(decoded)

# Save results
# df["generated_reasoning"] = results

results_df = DataFrame()
results_df['gold'] = df[INPUT_COLUMN]
results_df["predictions"] = results
results_df.to_excel("flan_t5_generated_reasoning.xlsx", sheet_name="data", index=False)
print("Inference complete. Output saved to flan_t5_generated_reasoning.xlsx")
