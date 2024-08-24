#!/bin/bash

python -m experiments.open_llms --model_name meta-llama/Llama-2-7b-chat-hf --visible_cuda_devices 0,1,2 --run_mode tag
python -m experiments.open_llms --model_name mistralai/Mistral-7B-Instruct-v0.3 --visible_cuda_devices 0,1,2 --run_mode tag
python -m experiments.open_llms --model_name microsoft/Phi-3-mini-128k-instruct --visible_cuda_devices 0,1,2 --run_mode tag
python -m experiments.open_llms --model_name Equall/Saul-7B-Instruct-v1 --visible_cuda_devices 0,1,2 --run_mode tag
python -m experiments.llama3 --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --visible_cuda_devices 0,1,2 --run_mode tag