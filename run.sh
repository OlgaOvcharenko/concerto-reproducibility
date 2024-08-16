#!/bin/bash

run_plot(){
    python make_plot.py --epoch 200 --lr 0.001 --batch_size 64 --drop_rate 0.1 --attention_s 0 --attention_t 1 --heads 16 --data simulated --train 0 --test 1 --model_type $1 --combine_omics 0 > $2 2>&1 &    
}

copy_data(){
    scp oovcharenko@euler.ethz.ch:/cluster/scratch/oovcharenko/concerto-reproducibility/Multimodal_pretraining/data/simulated/$1 Multimodal_pretraining/data/simulated/ &
}

run_scib(){
    nohup python metrics_cpu.py --data Multimodal_pretraining/data/$1/$2
    mv Multimodal_pretraining/plots/metrics/$1/scib_results.svg Multimodal_pretraining/plots/metrics/simulated/$3
}

run_scib simulated simulated_train_0_mt_2_bs_64_200_0.001_0.1_False_True_16.h5ad simulated_train_0_mt_2_bs_64_200_0.001_0.1_False_True_16.svg
run_scib human human_train_0_mt_2_bs_64_150_0.0001_0.1_False_True_64.h5ad human_train_0_mt_2_bs_64_150_0.0001_0.1_False_True_64.svg
run_scib human_cite human_cite_train_0_mt_2_bs_64_100_0.0001_0.1_False_True_16.h5ad human_cite_train_0_mt_2_bs_64_100_0.0001_0.1_False_True_16.svg
run_scib human_cite_raw human_cite_raw_train_0_mt_2_bs_64_100_0.0001_0.1_False_True_16.h5ad human_cite_raw_train_0_mt_2_bs_64_100_0.0001_0.1_False_True_16.svg

wait
