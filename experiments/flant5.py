import pandas as pd
from pandas import DataFrame
from transformers import pipeline
from tqdm import tqdm

# Configuration
MODEL_NAME = "google/flan-t5-xl"  # or flan-t5-xl, flan-ul2
INPUT_COLUMN = "background"  # or "background"
EXCEL_PATH = "data/test_data_extended.xlsx"
SHEET_NAME = "data"
OUTPUT_PATH = "flan_t5_pipeline_batched_reasoning.xlsx"
BATCH_SIZE = 4

# Load test data
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df = df.dropna(subset=[INPUT_COLUMN, "decision_label"])

# Format prompts
prompts = [
    f"Decision of the following case is {row['decision_label']}. Then explain the reasoning for the case: {row[INPUT_COLUMN]}"
    for _, row in df.iterrows()
]

# Load pipeline with batching support
pipe = pipeline("text2text-generation", model=MODEL_NAME, device_map='auto')  # change device if needed

# Batched inference
results = []
for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Running inference"):
    batch = prompts[i:i + BATCH_SIZE]
    outputs = pipe(batch, max_length=512, truncation=True)
    batch_results = [o['generated_text'] for o in outputs]
    results.extend(batch_results)

# Save results
# df["generated_reasoning"] = results
# df.to_excel(OUTPUT_PATH, index=False)
# print(f"Inference complete. Output saved to {OUTPUT_PATH}")


results_df = DataFrame()
results_df['gold'] = df[INPUT_COLUMN]
results_df["predictions"] = results
results_df.to_excel("flan_t5_generated_reasoning.xlsx", sheet_name="data", index=False)
print("Inference complete. Output saved to flan_t5_generated_reasoning.xlsx")
