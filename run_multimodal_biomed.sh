#!/bin/bash

#SBATCH -p gpu
#SBATCH -o logs/log-%j-multimodal.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=32:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G

mkdir -p logs

enable_modules
module load code-server
#module load gcc/8.2.0 python_gpu/3.9.9
module load StdEnv/2020 gcc/8.4.0
module load python/3.9.6

nvidia-smi

source "python_venv/bin/activate"

python3 multimodal_correct.py --epoch $1 --lr $2 --batch_size $3 --drop_rate $4 --attention_s $5 --attention_t $6 --heads $7 --data $8 --train $9
