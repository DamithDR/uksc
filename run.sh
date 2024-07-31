#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=00:60:00
#SBATCH --cpus-per-task=60
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d.dolamullage@lancaster.ac.uk

source /etc/profile
module add cuda/12.0

nbody -benchmark -numbodies=2500000 -numdevices=1

