#!/bin/bash
export HF_HOME="/mnt/data/dolamull/hf_cache"

echo "Hugging Face cache directory set to: $HF_HOME"

python -m experiments.lg.lg_exp --model meta-llama/Llama-2-7b-chat-hf --max-tokens 2048
python -m experiments.lg.lg_exp --model mistralai/Mistral-7B-Instruct-v0.3 --max-tokens 2048
python -m experiments.lg.lg_exp --model microsoft/Phi-3-mini-128k-instruct --max-tokens 2048
python -m experiments.lg.lg_exp --model Equall/Saul-7B-Instruct-v1 --max-tokens 2048
python -m experiments.lg.lg_exp --model meta-llama/Meta-Llama-3.1-8B-Instruct --max-tokens 2048