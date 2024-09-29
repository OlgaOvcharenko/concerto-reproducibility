#!/bin/bash

#SBATCH -o logs_cpu/log-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=2GB
#SBATCH --mail-type=END,FAIL

mkdir -p logs

conda activate venv

# nvidia-smi

# python3 spatial_multimodal_data.py --epoch $1 --lr $2 --batch_size $3 --drop_rate $4 --attention_s $5 --attention_t $6 --heads $7 --data $8 --train $9 --test ${10} --model_type ${11} --combine_omics ${12}
python3 prepare_PBMC.py --epoch $1 --lr $2 --batch_size $3 --drop_rate $4 --attention_s $5 --attention_t $6 --heads $7 --data $8 --train $9 --test ${10} --model_type ${11} --combine_omics ${12}
# python3 prepare_neurips_cite.py --epoch $1 --lr $2 --batch_size $3 --drop_rate $4 --attention_s $5 --attention_t $6 --heads $7 --data $8 --train $9 --test ${10} --model_type ${11} --combine_omics ${12}

#python3 metrics.py --data Multimodal_pretraining/data/human_cite/human_cite_bc_1_mt_0_bs_64_3_0.001_0.1_False_True_128_0.h5ad
# python3 metrics.py --data Multimodal_pretraining/data/simulated/simulated_bc_1_mt_0_bs_64_3_0.001_0.1_False_True_128_0.h5ad
