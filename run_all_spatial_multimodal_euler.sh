#!/bin/bash

mkdir -p logs

source "python_venv/bin/activate"

epochs=("200")
lrs=("1e-5")
batch_sizes=("1024")
drop_rates=("0.1")
attention_t=("1")
attention_s=("0")
heads=("64")
data=("spatial_split")
train=0
test=1
mask=("0 1")

model_type=("1 2")
combine_omics=0

# epochs=("200")
# lrs=("1e-5")
# batch_sizes=("1024")
# drop_rates=("0.1")
# attention_t=("1")
# attention_s=("0")
# heads=("64")
# data=("spatial_split")
# train=0
# test=1
# mask=("0")

# model_type=("1")
# combine_omics=0

for e in $epochs; do
    for lr in $lrs; do
        for batch_size in $batch_sizes; do
            for drop_rate in $drop_rates; do
                for s in $attention_s; do
                    for t in $attention_t; do
                        for h in $heads; do
                            for d in $data; do
                                for mt in $model_type; do
                                    for msk in $mask; do
                                        sbatch run_spatial_multimodal_euler.sh $e $lr $batch_size $drop_rate $s $t $h $d $train $test $mt $combine_omics $msk
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
