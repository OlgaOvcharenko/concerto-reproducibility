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

# epochs=("1")
# lrs=("1e-5")
# batch_sizes=("64")
# drop_rates=("0.1 0.4")
# attention_t=("1")
# attention_s=("0")
# heads=("64 128")
# data=("simulated human")

epochs=("500")
lrs=("1e-5 1e-3")
batch_sizes=("64")
drop_rates=("0.0 0.1 0.4")
attention_t=("1")
attention_s=("0")
heads=("64 128")
data=("simulated")

# epochs=("500")
# lrs=("1e-5")
# batch_sizes=("64")
# drop_rates=("0.1")
# attention_t=("1")
# attention_s=("0")
# heads=("128")
# data=("simulated")
train=0

#epochs=("256")
#lrs=("1e-5")
#batch_sizes=("64")
#drop_rates=("0.0")
#attention_t=("0")
#attention_s=("0")
#heads=("128")
#data=("simulated")

for e in $epochs; do
    for lr in $lrs; do
        for batch_size in $batch_sizes; do
            for drop_rate in $drop_rates; do
                for s in $attention_s; do
                    for t in $attention_t; do
                        for h in $heads; do
                            for d in $data; do
                                sbatch run_multimodal.sh $e $lr $batch_size $drop_rate $s $t $h $d $train
                            done
                        done
                    done
                done
            done
        done
    done 
done
