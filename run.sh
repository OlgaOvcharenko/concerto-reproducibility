#!/bin/bash

run_plot(){
    python make_plot.py --epoch 200 --lr 0.001 --batch_size 64 --drop_rate 0.1 --attention_s 0 --attention_t 1 --heads 16 --data simulated --train 0 --test 1 --model_type $1 --combine_omics 0 > $2 2>&1 &    
}

copy_data(){
    scp oovcharenko@euler.ethz.ch:/cluster/scratch/oovcharenko/concerto-reproducibility/Multimodal_pretraining/data/simulated/$1 Multimodal_pretraining/data/simulated/ &
}

run_scib(){
    nohup python metrics_cpu.py --data Multimodal_pretraining/data/simulated/$1
    mv Multimodal_pretraining/plots/metrics/simulated/scib_results.svg Multimodal_pretraining/plots/metrics/simulated/$2
}


copy_data simulated_train_1_mt_0_bs_64_200_0.001_0.1_False_True_16.h5ad
copy_data simulated_test_1_mt_0_bs_64_200_0.001_0.1_False_True_16.h5ad
copy_data simulated_train_0_mt_1_bs_64_200_0.001_0.1_False_True_16.h5ad
copy_data simulated_train_0_mt_2_bs_64_200_0.001_0.1_False_True_16.h5ad
copy_data simulated_train_0_mt_5_bs_64_200_0.001_0.1_False_True_16.h5ad
copy_data simulated_train_0_mt_3_bs_64_200_0.001_0.1_False_True_16.h5ad
copy_data simulated_train_0_mt_4_bs_64_200_0.001_0.1_False_True_16.h5ad
copy_data simulated_test_0_mt_1_bs_64_200_0.001_0.1_False_True_16.h5ad
copy_data simulated_test_0_mt_2_bs_64_200_0.001_0.1_False_True_16.h5ad
copy_data simulated_test_0_mt_5_bs_64_200_0.001_0.1_False_True_16.h5ad
copy_data simulated_test_0_mt_3_bs_64_200_0.001_0.1_False_True_16.h5ad
copy_data simulated_test_0_mt_4_bs_64_200_0.001_0.1_False_True_16.h5ad

wait

run_plot 5 simulated_test_0_mt_5_16_plots.out
run_plot 4 simulated_test_0_mt_4_16_plots.out
run_plot 3 simulated_test_0_mt_3_16_plots.out
run_plot 2 simulated_test_0_mt_2_16_plots.out
run_plot 1 simulated_test_0_mt_1_16_plots.out

wait

run_scib simulated_train_1_mt_0_bs_64_200_0.001_0.1_False_True_16.h5ad simulated_train_1_mt_0_bs_64_200_0.001_0.1_False_True_16.svg
run_scib simulated_test_1_mt_0_bs_64_200_0.001_0.1_False_True_16.h5ad simulated_test_1_mt_0_bs_64_200_0.001_0.1_False_True_16.svg
run_scib simulated_train_0_mt_1_bs_64_200_0.001_0.1_False_True_16.h5ad simulated_train_0_mt_1_bs_64_200_0.001_0.1_False_True_16.svg
run_scib simulated_train_0_mt_2_bs_64_200_0.001_0.1_False_True_16.h5ad simulated_train_0_mt_2_bs_64_200_0.001_0.1_False_True_16.svg
run_scib simulated_train_0_mt_5_bs_64_200_0.001_0.1_False_True_16.h5ad simulated_train_0_mt_5_bs_64_200_0.001_0.1_False_True_16.svg
run_scib simulated_train_0_mt_3_bs_64_200_0.001_0.1_False_True_16.h5ad simulated_train_0_mt_3_bs_64_200_0.001_0.1_False_True_16.svg
run_scib simulated_train_0_mt_4_bs_64_200_0.001_0.1_False_True_16.h5ad simulated_train_0_mt_4_bs_64_200_0.001_0.1_False_True_16.svg
run_scib simulated_test_0_mt_1_bs_64_200_0.001_0.1_False_True_16.h5ad simulated_test_0_mt_1_bs_64_200_0.001_0.1_False_True_16.svg
run_scib simulated_test_0_mt_2_bs_64_200_0.001_0.1_False_True_16.h5ad simulated_test_0_mt_2_bs_64_200_0.001_0.1_False_True_16.svg
run_scib simulated_test_0_mt_5_bs_64_200_0.001_0.1_False_True_16.h5ad simulated_test_0_mt_5_bs_64_200_0.001_0.1_False_True_16.svg
run_scib simulated_test_0_mt_3_bs_64_200_0.001_0.1_False_True_16.h5ad simulated_test_0_mt_3_bs_64_200_0.001_0.1_False_True_16.svg
run_scib simulated_test_0_mt_4_bs_64_200_0.001_0.1_False_True_16.h5ad simulated_test_0_mt_4_bs_64_200_0.001_0.1_False_True_16.svg

wait
