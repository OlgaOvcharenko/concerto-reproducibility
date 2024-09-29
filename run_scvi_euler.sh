#!/bin/bash

#SBATCH -o logs/log-%j-scvi-multimodal.out
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --mail-type=END,FAIL

mkdir -p logs

#module load gcc/8.2.0 python_gpu/3.9.9
module load stack/.2024-04-silent stack/2024-04
module load gcc/8.5.0
module --ignore_cache load python/3.9.18

conda activate venv

python3 scVI_multimodal.py --data simulated --epoch 100 --task 0 --train 1 --test 1
