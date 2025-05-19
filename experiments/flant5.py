import pandas as pd
from pandas import DataFrame
from transformers import pipeline

# Configuration
MODEL_NAME = "google/flan-t5-base"  # or flan-t5-xl, flan-ul2
INPUT_COLUMN = "judgment"  # or "background"
EXCEL_PATH = "data/test_data_extended.xlsx"
SHEET_NAME = "data"
OUTPUT_PATH = "flan_t5_pipeline_generated_reasoning.xlsx"

# Load test data
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df = df.dropna(subset=[INPUT_COLUMN, "decision_label"])

# Format prompts
prompts = [
    f"Decision of the following case is {row['decision_label']}. Then explain the reasoning for the case: {row[INPUT_COLUMN]}"
    for _, row in df.iterrows()
]

# Load the pipeline
pipe = pipeline("text2text-generation", model=MODEL_NAME, device_map='auto')  # Use device=0 for GPU or -1 for CPU

# Run inference
results = []
for prompt in prompts:
    output = pipe(prompt, max_length=256, truncation=True)[0]['generated_text']
    results.append(output)

# Save results
# df["generated_reasoning"] = results
# df.to_excel(OUTPUT_PATH, index=False)
# print(f"Inference complete. Output saved to {OUTPUT_PATH}")


results_df = DataFrame()
results_df['gold'] = df[INPUT_COLUMN]
results_df["predictions"] = results
results_df.to_excel("flan_t5_generated_reasoning.xlsx", sheet_name="data", index=False)
print("Inference complete. Output saved to flan_t5_generated_reasoning.xlsx")
