import gzip
import os
import tempfile
from pathlib import Path

import numpy as np
from concerto_function5_3 import preprocessing_changed_rna
import pooch
import scanpy as sc
import scvi
import seaborn as sns
import torch
import anndata as ad


def prepare_data_PBMC(adata_RNA, adata_Protein, train: bool = True, is_hvg_RNA: bool = True, is_hvg_protein: bool = False):
    print("Read PBMC data.")
    print(f"Train={train} RNA data shape {adata_RNA.shape}")
    print(f"Train={train} Protein data shape {adata_Protein.shape}")

    adata_RNA = preprocessing_changed_rna(adata_RNA, min_features=0, is_hvg=is_hvg_RNA, batch_key='batch')
    adata_Protein = preprocessing_changed_rna(adata_Protein, min_features=0, is_hvg=is_hvg_protein, batch_key='batch')
    
    # Add PCA after preprocessing for benchmarking
    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
    
    return adata_merged


# Settings
scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")
save_dir = tempfile.TemporaryDirectory()

# Data

path = './Multimodal_pretraining/data/multi_gene_l2.loom'
adata_RNA = sc.read(path)
path = './Multimodal_pretraining/data/multi_protein_l2.loom'
adata_Protein = sc.read(path) 

train_idx = (adata_RNA.obs["batch"] != "P6") & (adata_RNA.obs["batch"] != "P7") & (adata_RNA.obs["batch"] != "P8")
test_idx = (train_idx != 1)

adata_RNA_test = adata_RNA[test_idx, :]
adata_Protein_test = adata_Protein[test_idx, :]

adata_RNA = adata_RNA[train_idx, :]
adata_Protein = adata_Protein[train_idx, :]

# TODO Remove cell type B from reference
cell_ix = (adata_RNA.obs["cell_type"] != "B intermediate") & (adata_RNA.obs["cell_type"] != "B memory") & (adata_RNA.obs["cell_type"] != "B naive") & (adata_RNA.obs["cell_type"] != "Plasmablast")
adata_RNA = adata_RNA[cell_ix, :]
adata_Protein = adata_Protein[cell_ix, :]

save_path = './Multimodal_pretraining/'
adata_merged = prepare_data_PBMC(adata_RNA=adata_RNA, adata_Protein=adata_Protein, train=True, save_path=save_path, is_hvg_protein=True)
adata_merged_test = prepare_data_PBMC(adata_RNA=adata_RNA_test, adata_Protein=adata_Protein_test, train=False, save_path=save_path, is_hvg_protein=True)


l2tol1 = {
    'CD8 Naive': 'CD8 T',
    'CD8 Proliferating': 'CD8 T',
    'CD8 TCM': 'CD8 T',
    'CD8 TEM': 'CD8 T',
    'CD8+ T': 'CD8 T',
    'CD8+ T naive': 'CD8 T',
    'CD4 CTL': 'CD4 T',
    'CD4 Naive': 'CD4 T',
    'CD4 Proliferating': 'CD4 T',
    'CD4 TCM': 'CD4 T',
    'CD4 TEM': 'CD4 T',
    'CD4+ T activated': 'CD4 T',
    'CD4+ T naive': 'CD4 T',
    'CD14+ Mono': 'CD14 T',
    'CD16+ Mono': 'CD16 T',
    'Treg': 'CD4 T',
    'NK': 'NK',
    'NK Proliferating': 'NK',
    'NK_CD56bright': 'NK',
    'dnT': 'other T',
    'gdT': 'other T',
    'ILC': 'other T',
    'MAIT': 'other T',
    'CD14 Mono': 'Monocytes',
    'CD16 Mono': 'Monocytes',
    'cDC1': 'DC',
    'cDC2': 'DC',
    'pDC': 'DC',
    'ASDC':'DC',
    'B intermediate': 'B',
    'B memory': 'B',
    'B naive': 'B',
    'B1 B': 'B',
    'Plasmablast': 'B',
    'Eryth': 'other',
    'HSPC': 'other',
    'Platelet': 'other'
}

# if data == 'simulated':
adata_merged.obs['cell_type_l1'] = adata_merged.obs['cell_type'].map(l2tol1)
adata_merged_test.obs['cell_type_l1'] = adata_merged_test.obs['cell_type'].map(l2tol1)

adata_merged.var_names_make_unique()
adata_merged_test.var_names_make_unique()

# scvi.model.TOTALVI.setup_anndata(
#     adata_merged, batch_key="batch", protein_expression_obsm_key="protein_expression"
# )

model = scvi.model.TOTALVI(adata_merged, latent_distribution="normal", n_layers_decoder=2)
model.train()

TOTALVI_LATENT_KEY = "X_totalVI"
PROTEIN_FG_KEY = "protein_fg_prob"

adata_merged.obsm[TOTALVI_LATENT_KEY] = model.get_latent_representation()
adata_merged.obsm[PROTEIN_FG_KEY] = model.get_protein_foreground_probability(transform_batch="PBMC10k")

rna, protein = model.get_normalized_expression(
    transform_batch="PBMC10k", n_samples=25, return_mean=True
)

_, protein_means = model.get_normalized_expression(
    n_samples=25,
    transform_batch="PBMC10k",
    include_protein_background=True,
    sample_protein_mixing=False,
    return_mean=True,
)

TOTALVI_CLUSTERS_KEY = "leiden_totalVI"

sc.pp.neighbors(adata_merged, use_rep=TOTALVI_LATENT_KEY)
sc.tl.umap(adata_merged, min_dist=0.4)
sc.tl.leiden(adata_merged, key_added=TOTALVI_CLUSTERS_KEY)

perm_inds = np.random.permutation(len(adata_merged))
sc.pl.umap(
    adata_merged[perm_inds],
    color=[TOTALVI_CLUSTERS_KEY, "batch", "cell_type_l1"],
    ncols=1,
    frameon=False,
)