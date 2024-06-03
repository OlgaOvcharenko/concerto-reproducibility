#!/bin/bash

#SBATCH -o logs/log-%j-metrics.out
#SBATCH --nodes=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=03:00:00
#SBATCH --gres=gpumem:24G
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=END,FAIL

mkdir -p logs

module load gcc/8.2.0 python_gpu/3.9.9
nvidia-smi

source "metrics_python_venv/bin/activate"

for filename in ./Multimodal_pretraining/data/simulated/*.h5ad; do
    echo $filename
    python3 metrics.py --data $filename &
done

wait
