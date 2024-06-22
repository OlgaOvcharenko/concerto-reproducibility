import argparse
import os
import sys
sys.path.append("../")
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, silhouette_samples

from scib_metrics.benchmark import Benchmarker, BioConservation

import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

def get_args():
    parser = argparse.ArgumentParser(description='CONCERTO Batch Correction.')

    parser.add_argument('--data', type=str, required=True,
                        help='Adata path')

    args = parser.parse_args()
    return args


args = get_args()
data = args.data

print("Read adata")
adata_merged = sc.read_h5ad(data)
print(adata_merged)


print("Start metrics")
names_obs = list(adata_merged.obsm.keys())
names_obs.remove("X_pca")
biocons = BioConservation(isolated_labels=True, nmi_ari_cluster_labels_leiden=True, nmi_ari_cluster_labels_kmeans=False)

start = time.time()
bm = Benchmarker(
    adata_merged,
    batch_key="batch",
    label_key="cell_type",
    embedding_obsm_keys=names_obs,
    bio_conservation_metrics=biocons,
    n_jobs=-1,
)

bm.benchmark()
end = time.time()
print(f"Time: {int((end - start) / 60)} min {int((end - start) % 60)} sec")

df = bm.get_results(min_max_scale=False)
print(df)
df = df.round(decimals=3)
df.to_csv(f'./Multimodal_pretraining/plots/metrics/{data.split("/")[-1][:-5]}_metrics.csv')

# bm.plot_results_table(save_dir=f'./Multimodal_pretraining/plots/metrics/')
# df.to_csv(f'./Multimodal_pretraining/plots/metrics/{data[:-5]}')

dir = f"{data[:-5]}"
# data.split("/")[-1][:-5]
if not os.path.exists(dir):
    os.makedirs(dir)
bm.plot_results_table(save_dir='./Multimodal_pretraining/plots/metrics/')

