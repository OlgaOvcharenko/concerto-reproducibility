#!/bin/bash

mkdir -p logs

source "python_venv/bin/activate"

epochs=("10 64 128 256 512")
lrs=("1e-2 1e-5")
batch_sizes=("64")
drop_rate=("0.0 0.1 0.2 0.4")

for e in $epochs; do
    for lr in $lrs; do
        for batch_size in $batch_sizes; do
        sbatch run_batch_correct.ah $e $lr $batch_size $drop_rate
    done 
done
