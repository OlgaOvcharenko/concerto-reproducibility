#!/bin/bash

mkdir -p logs

source "python_venv/bin/activate"

# epochs=("10 64 128 256 512")
# lrs=("1e-2 1e-5")
# batch_sizes=("64")
# drop_rates=("0.0 0.1 0.4")
# attention_t=("True False")
# attention_s=("True False")
# heads=("128")

# epochs=("32 128")
# lrs=("1e-5")
# batch_sizes=("64")
# drop_rates=("0.1 0.4")
# attention_t=("True")
# attention_s=("False")
# heads=("64 128")

epochs=("64")
lrs=("1e-5")
batch_sizes=("64")
drop_rates=("0.1")
attention_t=("1")
attention_s=("0")
heads=("64")

for e in $epochs; do
    for lr in $lrs; do
        for batch_size in $batch_sizes; do
            for drop_rate in $drop_rates; do
                for s in $attention_s; do
                    for t in $attention_t; do
                        for h in $heads; do
                            sbatch run_batch_correct.sh $e $lr $batch_size $drop_rate $s $t $h
                        done
                    done
                done
            done
        done
    done 
done
