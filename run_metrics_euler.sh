#!/bin/bash

#SBATCH -o logs/log-%j-metrics.out
#SBATCH --nodes=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=04:00:00
#SBATCH --gres=gpumem:24G
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=END,FAIL

mkdir -p logs

module load gcc/8.2.0 python_gpu/3.9.9 cuda/11.3.1
nvidia-smi

source "metrics_python_venv/bin/activate"

echo $1
python3 metrics.py --data $1
