#!/bin/bash

#SBATCH -o logs/log-%j-scvi-multimodal.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --mail-type=END,FAIL

mkdir -p logs

#module load gcc/8.2.0 python_gpu/3.9.9
# module load stack/.2024-04-silent stack/2024-04
# module load gcc/8.5.0
# module --ignore_cache load python/3.9.18

conda activate myenv

# python3 scVI_multimodal.py --data simulated --epoch 100 --task 0 --train 1 --test 1
# python3 scVI_multimodal.py --data human_cite --epoch 100 --task 0 --train 1 --test 1
# python3 multiVI_multimodal.py --data human_multiome --epoch 100 --task 0 --train 1 --test 1

# python3 scVI_multimodal.py --data simulated --epoch 100 --task 1 --train 1 --test 1
# python3 scVI_multimodal.py --data human_cite --epoch 100 --task 1 --train 1 --test 1
python3 multiVI_multimodal.py --data human_multiome --epoch 100 --task 1 --train 1 --test 1
