import argparse
import os
from typing import List, Tuple

import pandas as pd
from graphviz import Digraph
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.lg.HuggingFaceLLM import HuggingFaceLLM
from experiments.lg.JudgmentDataset import JudgmentDataset
from experiments.lg.JudgmentState import JudgmentState
from experiments.lg.util import process_chunk, predict_judgment


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


# Main function to set up and run the graph for a single text
def run_judgment_predictor(judgment_text: str, llm: HuggingFaceLLM, max_tokens: int = 2048) -> Tuple[str, str]:
    # Initialize the state with chunks as a list
    chunks = chunk_text_by_tokens(judgment_text, max_tokens, llm.tokenizer)
    state: JudgmentState = {
        "chunks_processed": [],
        "full_text_summary": "",
        "judgment_prediction": None,
        "chunks": chunks,
        "current_chunk_idx": 0
    }

    # loop until all chunks are finished
    while state['current_chunk_idx'] < len(state['chunks_processed']):
        state = process_chunk(llm, state, max_tokens)

    final_state = predict_judgment(llm, state, max_tokens)

    return final_state["judgment_prediction"], final_state["full_text_summary"]


# Load judgment text and ground truth from Excel file
def load_dataset(file_path: str = "data/UKSC_dataset_extended.xlsx") -> pd.DataFrame:
    df = pd.read_excel(file_path)

    df = df[:8]
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


# Function to visualize the graph
def visualize_langgraph(graph):
    dot = Digraph(comment="LangGraph Visualization")
    dot.attr(rankdir="LR")  # Left-to-right layout

    # Add nodes
    for node in graph.nodes:
        dot.node(node, label=node)

    # Add edges
    for edge in graph.edges:
        start, end = edge
        dot.edge(start, end)

    # Add END as terminating node
    dot.node("END", shape="doublecircle")

    # Render and display
    dot.render("langgraph_output", view=True, format="png")  # Saves as PNG and opens it


# Run the script
if __name__ == "__main__":
    main()
