from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, List, Optional
import asyncio
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader


# Define the state structure to hold accumulated information
class JudgmentState(TypedDict):
    chunks_processed: List[str]  # Processed summaries of each chunk
    full_text_summary: str  # Running summary of the entire text
    judgment_prediction: Optional[str]  # Final prediction (allow/dismiss)


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
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Wrap model with DataParallel for multi-GPU
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()

    async def agenerate(self, prompts: List[str]) -> List[dict]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.module.generate(**inputs, max_new_tokens=100, do_sample=False) if isinstance(
                self.model, nn.DataParallel) else self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return [{"text": output[len(prompt):].strip()} for prompt, output in zip(prompts, decoded_outputs)]


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


# Process a single chunk of text
async def process_chunk(state: JudgmentState, chunk: str, llm: HuggingFaceLLM) -> JudgmentState:
    prompt = PromptTemplate(
        input_variables=["chunk", "current_summary"],
        template="Given the following chunk of a legal judgment: '{chunk}', and the current summary of previous chunks: '{current_summary}', provide a concise summary of this chunk and integrate it into the overall summary."
    )
    response = await llm.agenerate(
        [prompt.format(chunk=chunk, current_summary=state["full_text_summary"] or "No summary yet.")])
    chunk_summary = response[0]["text"]
    state["chunks_processed"].append(chunk_summary)
    state["full_text_summary"] = chunk_summary  # Update running summary
    return state


# Predict judgment based on the full summary
async def predict_judgment(state: JudgmentState, llm: HuggingFaceLLM) -> JudgmentState:
    prompt = PromptTemplate(
        input_variables=["summary"],
        template="Based on the following summary of a legal judgment: '{summary}', predict the outcome as either 'allow' or 'dismiss'. Provide a single-word answer."
    )
    response = await llm.agenerate([prompt.format(summary=state["full_text_summary"])])
    prediction = response[0]["text"].lower()
    state["judgment_prediction"] = "allow" if prediction == "allow" else "dismiss"
    return state


# Main function to set up and run the graph for a single text
async def run_judgment_predictor(judgment_text: str, llm: HuggingFaceLLM, max_tokens: int = 1000):
    initial_state: JudgmentState = {
        "chunks_processed": [],
        "full_text_summary": "",
        "judgment_prediction": None
    }

    workflow = StateGraph(JudgmentState)
    workflow.add_node("process_chunk", lambda state: process_chunk(state, next(state["chunks"]), llm))
    workflow.add_node("predict_judgment", lambda state: predict_judgment(state, llm))

    workflow.set_entry_point("process_chunk")
    chunks = chunk_text_by_tokens(judgment_text, max_tokens, llm.tokenizer)
    initial_state["chunks"] = iter(chunks)

    def should_continue(state):
        try:
            next(state["chunks"])
            return "process_chunk"
        except StopIteration:
            return "predict_judgment"

    workflow.add_conditional_edges("process_chunk", should_continue)
    workflow.add_edge("predict_judgment", END)

    app = workflow.compile()
    final_state = await app.ainvoke(initial_state)
    return final_state["judgment_prediction"]


# Load judgment text and ground truth from Excel file
def load_dataset(file_path: str = "data/UKSC_dataset_extended.xlsx") -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df = df[:10] # todo remove after testing
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


# Process dataset in batches
async def process_in_batches(dataset: JudgmentDataset, llm: HuggingFaceLLM, max_tokens: int, batch_size: int):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    true_labels = []

    for batch_texts, batch_labels in dataloader:
        batch_predictions = []
        for text in batch_texts:
            prediction = await run_judgment_predictor(text, llm, max_tokens)
            batch_predictions.append(prediction)
        predictions.extend(batch_predictions)
        true_labels.extend(batch_labels)
        print(f"Processed batch: {len(batch_predictions)} samples")

    return predictions, true_labels


# Main function
async def main():
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

    # Context lengths and batch sizes for specified models (batch size capped at 8)
    model_configs = {
        "meta-llama/Llama-2-7b-chat-hf": {"context_length": 4096, "batch_size": 8},
        "mistralai/Mistral-7B-Instruct-v0.3": {"context_length": 32768, "batch_size": 8},
        "microsoft/Phi-3-mini-128k-instruct": {"context_length": 128000, "batch_size": 8},
        "Equall/Saul-7B-Instruct-v1": {"context_length": 32768, "batch_size": 8},
        "meta-llama/Meta-Llama-3.1-8B-Instruct": {"context_length": 128000, "batch_size": 8}
    }
    config = model_configs.get(args.model, {"context_length": 4096, "batch_size": 8})
    max_tokens = min(args.max_tokens, config["context_length"] // 4)
    batch_size = config["batch_size"]

    # Process in batches
    predictions, true_labels = await process_in_batches(dataset, llm, max_tokens, batch_size)

    # Compute and save metrics
    metrics = compute_metrics(true_labels, predictions)
    save_results(args.model, metrics, predictions, true_labels)


# Run the script
import platform

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())