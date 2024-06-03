#!/bin/bash

mkdir -p logs

for filename in ./Multimodal_pretraining/data/simulated/*.h5ad; do
    echo $filename
    sbatch run_metrics_euler.sh $filename
done
