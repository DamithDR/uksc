#!/bin/bash

export HF_HOME="/mnt/data/dolamull/hf_cache"

echo "Hugging Face cache directory set to: $HF_HOME"

python -m experiments.llama3 --input_column judgment --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --visible_cuda_devices 0,1,2
