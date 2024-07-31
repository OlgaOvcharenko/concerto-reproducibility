#!/bin/bash

mkdir -p logs

source "python_venv/bin/activate"

epochs=("2")
lrs=("1e-3")
batch_sizes=("64")
drop_rates=("0.1")
attention_t=("1")
attention_s=("0")
heads=("16")
data=("simulated")
train=0
test=1

model_type=("1")
combine_omics=0

for e in $epochs; do
    for lr in $lrs; do
        for batch_size in $batch_sizes; do
            for drop_rate in $drop_rates; do
                for s in $attention_s; do
                    for t in $attention_t; do
                        for h in $heads; do
                            for d in $data; do
                                for mt in $model_type; do
                                    sbatch run_multimodal_euler.sh $e $lr $batch_size $drop_rate $s $t $h $d $train $test $mt $combine_omics
                                done
                            done
                        done
                    done
                done
            done
        done
    done 
done
