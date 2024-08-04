#!/bin/bash

run_plot(){
    python make_plot.py --epoch 200 --lr 0.001 --batch_size 64 --drop_rate 0.1 --attention_s 0 --attention_t 1 --heads 16 --data simulated --train 0 --test 1 --model_type $1 --combine_omics 0 > $2 2>&1 &    
}

run_plot 5 simulated_test_0_mt_5_16_plots.out
run_plot 4 simulated_test_0_mt_4_16_plots.out
run_plot 3 simulated_test_0_mt_3_16_plots.out
run_plot 2 simulated_test_0_mt_2_16_plots.out
run_plot 1 simulated_test_0_mt_1_16_plots.out

# python make_plot.py --epoch 200 --lr 0.001 --batch_size 64 --drop_rate 0.1 --attention_s 0 --attention_t 1 --heads 16 --data simulated --train 0 --test 1 --model_type 5 --combine_omics 0 > simulated_test_0_mt_5_16_plots.out 2>&1 &
# python make_plot.py --epoch 200 --lr 0.001 --batch_size 64 --drop_rate 0.1 --attention_s 0 --attention_t 1 --heads 16 --data simulated --train 0 --test 1 --model_type 4 --combine_omics 0 > simulated_test_0_mt_4_16_plots.out 2>&1 &
# python make_plot.py --epoch 200 --lr 0.001 --batch_size 64 --drop_rate 0.1 --attention_s 0 --attention_t 1 --heads 16 --data simulated --train 0 --test 1 --model_type 3 --combine_omics 0 > simulated_test_0_mt_3_16_plots.out 2>&1 &
# python make_plot.py --epoch 200 --lr 0.001 --batch_size 64 --drop_rate 0.1 --attention_s 0 --attention_t 1 --heads 16 --data simulated --train 0 --test 1 --model_type 2 --combine_omics 0 > simulated_test_0_mt_2_16_plots.out 2>&1 &
# python make_plot.py --epoch 200 --lr 0.001 --batch_size 64 --drop_rate 0.1 --attention_s 0 --attention_t 1 --heads 16 --data simulated --train 0 --test 1 --model_type 1 --combine_omics 0 > simulated_test_0_mt_1_16_plots.out 2>&1 &

wait
