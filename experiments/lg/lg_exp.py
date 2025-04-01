import argparse
import asyncio
import os
from typing import TypedDict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# Define the state structure to hold accumulated information
class JudgmentState(TypedDict):
    chunks_processed: List[str]  # Processed summaries of each chunk
    full_text_summary: str  # Running summary of the entire text
    judgment_prediction: Optional[str]  # Final prediction (allow/dismiss)
    chunks: List[str]  # List of chunks to process
    current_chunk_idx: int  # Index of the current chunk being processed


# Custom dataset for batching
class JudgmentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# Custom LLM wrapper for Hugging Face models with multi-GPU support
class HuggingFaceLLM:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Wrap model with DataParallel for multi-GPU
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()

    async def agenerate(self, prompts: List[str], max_length: int) -> List[dict]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(
            self.device)
        with torch.no_grad():
            outputs = self.model.module.generate(**inputs, max_new_tokens=2048, do_sample=True) if isinstance(
                self.model, nn.DataParallel) else self.model.generate(**inputs, max_new_tokens=2048, do_sample=True)
        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return [{"text": output[len(prompt):].strip()} for prompt, output in zip(prompts, decoded_outputs)]

    def generate(self, prompts: List[str], max_length: int) -> List[dict]:
        return asyncio.run(self.agenerate(prompts, max_length))


# Function to chunk text based on token count using the model's tokenizer
def chunk_text_by_tokens(text: str, max_tokens: int, tokenizer) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for token in tokens:
        if current_token_count + 1 > max_tokens:
            chunks.append(tokenizer.decode(current_chunk, skip_special_tokens=True))
            current_chunk = [token]
            current_token_count = 1
        else:
            current_chunk.append(token)
            current_token_count += 1

    if current_chunk:
        chunks.append(tokenizer.decode(current_chunk, skip_special_tokens=True))

    return chunks


# Process a single chunk of text (synchronous)
def process_chunk(state: JudgmentState, llm: HuggingFaceLLM, max_length: int) -> JudgmentState:
    chunk = state["chunks"][state["current_chunk_idx"]]

    is_final = state['current_chunk_idx'] == len(state['chunks']) - 1

    prompt = PromptTemplate(
        input_variables=["chunk", "current_summary"],
        template="Given the following chunk of a legal judgment: '{chunk}', and the current summary of all previous chunks: "
                 "'{current_summary}', provide a concise summary of this chunk and then generate an overall summary for all the chunks given upto now."
        if is_final else "Given the final chunk of this legal judgment: '{chunk}', and the current summary of all previous chunks: "
                         "'{current_summary}', provide a concise summary of this last chunk and use that to provide a final summary of the whole judgment."
    )

    response = llm.generate(
        [prompt.format(chunk=chunk, current_summary=state["full_text_summary"] or "No summary yet.")], max_length)
    chunk_summary = response[0]["text"]
    state["chunks_processed"].append(chunk_summary)
    state["full_text_summary"] = chunk_summary  # Update running summary
    state["current_chunk_idx"] += 1  # Move to the next chunk
    return state


# Predict judgment based on the full summary (synchronous)
def predict_judgment(state: JudgmentState, llm: HuggingFaceLLM, max_length: int) -> JudgmentState:
    prompt = PromptTemplate(
        input_variables=["summary"],
        # template="Based on the following summary of a legal judgment: '{summary}', predict the outcome as either "
        #          "'allow' or 'dismiss'. Provide a single-word answer.",
        template="""Assume you are a judge at the supreme court in United Kingdom. 
                    You will be provided UK supreme court appeal cases by the users and your duty is to understand the case background and output your decision label. 
                    Classify whether the provided appeal is allowed or dismissed, select one from following : [allow,dismiss].
                    Following is the summary of the judgment, please respond allow/dismiss, do not respond any explanation, other than allow/dismiss.
                    Summary : {summary}"""
    )
    response = llm.generate([prompt.format(summary=state["full_text_summary"])], max_length)
    prediction = response[0]["text"].lower()
    state["judgment_prediction"] = "allow" if prediction == "allow" else "dismiss"
    return state


# Main function to set up and run the graph for a single text
def run_judgment_predictor(judgment_text: str, llm: HuggingFaceLLM, max_tokens: int = 2048) -> Tuple[str, str]:
    # Initialize the state with chunks as a list
    chunks = chunk_text_by_tokens(judgment_text, max_tokens, llm.tokenizer)
    initial_state: JudgmentState = {
        "chunks_processed": [],
        "full_text_summary": "",
        "judgment_prediction": None,
        "chunks": chunks,
        "current_chunk_idx": 0
    }

    workflow = StateGraph(JudgmentState)
    workflow.add_node("process_chunk", lambda state: process_chunk(state, llm, max_tokens))
    workflow.add_node("predict_judgment", lambda state: predict_judgment(state, llm, max_tokens))

    workflow.set_entry_point("process_chunk")

    def should_continue(state):
        if state["current_chunk_idx"] < len(state["chunks"]):
            return "process_chunk"
        return "predict_judgment"

    workflow.add_conditional_edges("process_chunk", should_continue)
    workflow.add_edge("predict_judgment", END)

    app = workflow.compile()
    final_state = app.invoke(initial_state)
    return final_state["judgment_prediction"], final_state["full_text_summary"]


# Load judgment text and ground truth from Excel file
def load_dataset(file_path: str = "data/UKSC_dataset_extended.xlsx") -> pd.DataFrame:
    df = pd.read_excel(file_path)
    if "judgment_text" not in df.columns or "decision_label" not in df.columns:
        raise ValueError("Excel file must contain 'judgment_text' and 'decision_label' columns.")
    return df[["judgment_text", "decision_label"]].dropna()


# Compute evaluation metrics
def compute_metrics(true_labels: List[str], pred_labels: List[str]) -> dict:
    accuracy = accuracy_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    return {"Accuracy": accuracy, "Recall": recall, "Macro F1": macro_f1}


# Save results to a file
def save_results(model_name: str, metrics: dict, predictions: List[str], true_labels: List[str]):
    safe_model_name = model_name.replace("/", "_")
    file_name = f"{safe_model_name}_results.txt"
    with open(file_name, "w") as f:
        f.write("Evaluation Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\nPredictions vs Ground Truth:\n")
        for pred, true in zip(predictions, true_labels):
            f.write(f"Predicted: {pred}, True: {true}\n")
    print(f"Results saved to {file_name}")


# Save predictions and summaries to an Excel file
def save_outputs(model_name: str, predictions: List[str], summaries: List[str], true_labels: List[str]):
    safe_model_name = model_name.replace("/", "_")
    output_file = "model_outputs.xlsx"
    df = pd.DataFrame({
        "Prediction": predictions,
        "Full_Summary": summaries,
        "True_Label": true_labels
    })

    # Check if the file exists to determine the mode (write or append)
    mode = 'w' if not os.path.exists(output_file) else 'a'
    # with pd.ExcelWriter(output_file, engine='openpyxl', mode=mode, if_sheet_exists='replace') as writer:
    #     df.to_excel(writer, sheet_name=safe_model_name, index=False)
    #
    if not os.path.exists(output_file):
        df.to_excel(output_file, sheet_name=safe_model_name, index=False)
    else:
        with pd.ExcelWriter(output_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=safe_model_name, index=False)
    print(f"Outputs for {model_name} saved to {output_file} in sheet {safe_model_name}")


# Process dataset in batches
def process_in_batches(dataset: JudgmentDataset, llm: HuggingFaceLLM, max_tokens: int, batch_size: int):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    summaries = []
    true_labels = []

    for batch_texts, batch_labels in tqdm(dataloader):
        batch_predictions = []
        batch_summaries = []
        for text in batch_texts:
            prediction, summary = run_judgment_predictor(text, llm, max_tokens)
            batch_predictions.append(prediction)
            batch_summaries.append(summary)
        predictions.extend(batch_predictions)
        summaries.extend(batch_summaries)
        true_labels.extend(batch_labels)
        print(f"Processed batch: {len(batch_predictions)} samples")

    return predictions, summaries, true_labels


# Main function
def main():
    parser = argparse.ArgumentParser(description="Legal Judgment Predictor")
    parser.add_argument("--model", type=str, required=True,
                        help="Hugging Face model name (e.g., 'mistralai/Mistral-7B-Instruct-v0.3')")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max tokens per chunk")
    args = parser.parse_args()

    # Load the dataset
    df = load_dataset()
    dataset = JudgmentDataset(df["judgment_text"].astype(str).tolist(),
                              df["decision_label"].astype(str).str.lower().tolist())
    llm = HuggingFaceLLM(args.model)

    # Context lengths and batch sizes for specified models
    model_configs = {
        "meta-llama/Llama-2-7b-chat-hf": {"context_length": 4096, "batch_size": 12},
        "mistralai/Mistral-7B-Instruct-v0.3": {"context_length": 32768, "batch_size": 12},
        "microsoft/Phi-3-mini-128k-instruct": {"context_length": 128000, "batch_size": 18},
        "Equall/Saul-7B-Instruct-v1": {"context_length": 32768, "batch_size": 12},
        "meta-llama/Meta-Llama-3.1-8B-Instruct": {"context_length": 128000, "batch_size": 9}
    }
    config = model_configs.get(args.model, {"context_length": 4096, "batch_size": 12})
    max_tokens = min(args.max_tokens, config["context_length"] // 4)
    batch_size = config["batch_size"]

    # Process in batches
    predictions, summaries, true_labels = process_in_batches(dataset, llm, max_tokens, batch_size)

    # Save predictions and summaries to Excel
    save_outputs(args.model, predictions, summaries, predictions)

    # Compute and save metrics
    metrics = compute_metrics(true_labels, predictions)
    save_results(args.model, metrics, predictions, true_labels)


# Run the script
if __name__ == "__main__":
    main()
