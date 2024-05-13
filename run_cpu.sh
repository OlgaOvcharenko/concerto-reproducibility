#!/bin/bash

#SBATCH -o logs_cpu/log-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

mkdir -p logs

module load python

nvidia-smi

source "python_venv/bin/activate"

python3 $1

