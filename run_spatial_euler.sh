#!/bin/bash

#SBATCH -o logs_cpu/log-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --mail-type=END,FAIL

mkdir -p logs

module load stack/.2024-04-silent stack/2024-04
module load gcc/8.5.0
module --ignore_cache load python/3.9.18
source "python_venv/bin/activate"

# nvidia-smi

python3 spatial_multimodal.py --epoch $1 --lr $2 --batch_size $3 --drop_rate $4 --attention_s $5 --attention_t $6 --heads $7 --data $8 --train $9 --test ${10} --model_type ${11} --combine_omics ${12}
