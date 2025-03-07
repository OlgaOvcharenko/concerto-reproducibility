#!/bin/bash

#SBATCH -o logs/log-%j-multimodal.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpumem:8G
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END,FAIL

mkdir -p logs

nvidia-smi

conda activate concerto

python3 check_gpu.py 
