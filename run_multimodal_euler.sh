#!/bin/bash

#SBATCH -o logs/log-%j-multimodal.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --gres=gpumem:23G
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --mail-type=END,FAIL

mkdir -p logs

#module load gcc/8.2.0 python_gpu/3.9.9
module load stack/.2024-04-silent stack/2024-04
module load gcc/8.5.0
module --ignore_cache load python/3.9.18
nvidia-smi

source "python_venv/bin/activate"

python3 multimodal_correct.py --epoch $1 --lr $2 --batch_size $3 --drop_rate $4 --attention_s $5 --attention_t $6 --heads $7 --data $8 --train $9 --test ${10} --model_type ${11} --combine_omics ${12}
