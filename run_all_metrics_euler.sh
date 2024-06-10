#!/bin/bash

mkdir -p logs

for filename in ./Multimodal_pretraining/data/simulated/*.h5ad; do
    echo $filename
    sbatch run_metrics_euler.sh $filename
done

#sbatch run_metrics_euler.sh ./Multimodal_pretraining/data/simulated/simulated_403_1e-05_0.0_True_True_64.h5ad

