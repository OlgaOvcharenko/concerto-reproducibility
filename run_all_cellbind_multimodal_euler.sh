#!/bin/bash

mkdir -p logs

source "python_venv/bin/activate"

batch_sizes=("256")
batch_sizes2=("256")
drop_rates=("0.1 0.3")
attention_t=("1")
attention_s=("0")
heads=("64 128 256")
train=1
test=1

epochs=("150")
lrs=("1e-5")
data=("human")
model_type=("2")
combine_omics=1
tasks=("1")


for e in $epochs; do
    for lr in $lrs; do
        for batch_size in $batch_sizes; do
            for drop_rate in $drop_rates; do
                for s in $attention_s; do
                    for t in $attention_t; do
                        for h in $heads; do
                            for d in $data; do
                                for mt in $model_type; do
                                    for task in $tasks; do
                                        for bs2 in $batch_sizes2; do
                                            sbatch run_cellbind_multimodal_euler.sh $e $lr $batch_size $drop_rate $s $t $h $d $train $test $mt $combine_omics $task $bs2
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done 
done
