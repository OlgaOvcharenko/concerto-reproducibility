#!/bin/bash

#SBATCH -o logs/log-%j-metrics.out
#SBATCH --nodes=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=05:00:00
#SBATCH --gres=gpumem:21G
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=END,FAIL

mkdir -p logs

conda activate venv

# python3 metrics.py --data Multimodal_pretraining/data/human_cite/human_cite_bc_1_mt_0_bs_64_3_0.001_0.1_False_True_128_0.h5ad
python3 metrics.py --data Multimodal_pretraining/data/simulated/simulated_bc_1_mt_0_bs_64_3_0.001_0.1_False_True_128_0.h5ad