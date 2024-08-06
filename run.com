#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d.dolamullage@lancaster.ac.uk
#SBATCH --output=/storage/hpc/41/dolamull/experiments/uksc/output.log
#SBATCH --error=/storage/hpc/41/dolamull/experiments/uksc/error.log

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/41/dolamull/conda_envs/llm_env
export HF_HOME=/scratch/hpc/41/dolamull/hf_cache

python -m experiments.llama2_exp --model_name meta-llama/Meta-Llama-3-8B-Instruct