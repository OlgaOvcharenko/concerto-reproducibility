import argparse
import os
import torch
import numpy as np
import pandas as pd
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


_BIO_METRICS = BioConservation(isolated_labels=True, 
                               nmi_ari_cluster_labels_leiden=True, 
                               nmi_ari_cluster_labels_kmeans=False, 
                               silhouette_label=True, 
                               clisi_knn=True
                               )
_BATCH_METRICS = BatchCorrection(graph_connectivity=True, 
                                 kbet_per_label=True, 
                                 ilisi_knn=True, 
                                 pcr_comparison=True, 
                                 silhouette_batch=True
                                 )


def get_args():
    parser = argparse.ArgumentParser(description='CONCERTO Batch Correction.')

    parser.add_argument('--data', type=str, required=True,
                        help='Adata path')

    args = parser.parse_args()
    return args


def evaluate_model(adata, batch_key="batch", cell_type_label="cell_type_l1"):
    names_obs = list(adata.obsm.keys())
    names_obs.remove("X_pca")
    names_obs.remove("Unintegrated_HVG_only")
    
    bm = Benchmarker(
                adata,
                batch_key=batch_key,
                label_key=cell_type_label,
                embedding_obsm_keys=names_obs,
                bio_conservation_metrics=_BIO_METRICS,
                batch_correction_metrics=_BATCH_METRICS,
                n_jobs=4,
            )
    bm.benchmark()
    a = bm.get_results(False, True)
    results = a[:1].astype(float).round(4)
    return results

args = get_args()
data = args.data

print("Read adata")
adata = None
for repeat in range(0, 5):
    file_repeat = data.split(".")[0][:-1] + f"{repeat}.h5ad"
    if repeat == 0:
        adata = sc.read_h5ad(file_repeat) 
    else:
        tmp = sc.read_h5ad(file_repeat) 
        names_obs = list(tmp.obsm.keys())
        names_obs.remove("X_pca")
        names_obs.remove("Unintegrated_HVG_only")
        adata.obsm[f"Embedding_{repeat}"] = tmp.obsm[names_obs[0]]

df = evaluate_model(adata=adata)
df.to_csv(f'./Multimodal_pretraining/results/{data.split("/")[-1][:-5].split("_")[0]}/{data.split("/")[-1][:-5]}_metrics_True.csv')