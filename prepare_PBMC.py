import os
import sys

import pandas as pd
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import issparse
sys.path.append("../")
import numpy as np
import scanpy as sc
import anndata as ad
import muon as mu

l2tol1 = {
 'CD8 Naive': 'CD8 T',
 'CD8 Proliferating': 'CD8 T',
 'CD8 TCM': 'CD8 T',
 'CD8 TEM': 'CD8 T',
 'CD4 CTL': 'CD4 T',
 'CD4 Naive': 'CD4 T',
 'CD4 Proliferating': 'CD4 T',
 'CD4 TCM': 'CD4 T',
 'CD4 TEM': 'CD4 T',
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
 'Plasmablast': 'B',
 'Eryth': 'other',
 'HSPC': 'other',
 'Platelet': 'other'
}

def preprocess_rna(
        adata,
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = 10000,
        n_top_features=2000,  # or gene list
        chunk_size: int = 20000,
        is_hvg = True,
        batch_key = 'batch',
        log=True
):
    if min_features is None: min_features = 600
    if n_top_features is None: n_top_features = 40000

    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    adata = adata[:, [gene for gene in adata.var_names
                      if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]

    cells_subset, _ = sc.pp.filter_cells(adata, min_genes=min_features)
    print(cells_subset)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # mu.prot.pp.dsb(mdata, mdata_raw, isotype_controls=isotype_controls)
    
    if is_hvg == True:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key=batch_key, inplace=True, subset=True)

    print('Processed dataset shape: {}'.format(adata.shape))
    return adata, cells_subset

def prepare_data_PBMC_together(train_idx, test_idx, adata_RNA, adata_Protein, train: bool = True, save_path: str = '', is_hvg_RNA: bool = True, is_hvg_protein: bool = False):
    print("Read PBMC data.")
    print(f"Train={train} RNA data shape {adata_RNA.shape}")
    print(f"Train={train} Protein data shape {adata_Protein.shape}")

    # Create PCA for benchmarking
    adata_RNA, cells_subset = preprocess_rna(adata_RNA,min_features = 0, is_hvg=is_hvg_RNA, batch_key='batch')
    adata_Protein = adata_Protein[:, cells_subset]

    adata_RNA.obs['cell_type_l1'] = adata_RNA.obs['cell_type'].map(l2tol1)
    adata_Protein.obs['cell_type_l1'] = adata_Protein.obs['cell_type'].map(l2tol1)
    
    adata_RNA_test = adata_RNA[test_idx, :]
    adata_Protein_test = adata_Protein[test_idx, :]

    adata_RNA = adata_RNA[train_idx, :]
    adata_Protein = adata_Protein[train_idx, :]

    cell_ix = (adata_RNA.obs["cell_type"] != "B intermediate") & (adata_RNA.obs["cell_type"] != "B memory") & (adata_RNA.obs["cell_type"] != "B naive") & (adata_RNA.obs["cell_type"] != "Plasmablast")
    adata_RNA = adata_RNA[cell_ix, :]
    adata_Protein = adata_Protein[cell_ix, :]

    adata_RNA.write_h5ad(save_path + f'adata_RNA_train.h5ad')
    adata_Protein.write_h5ad(save_path + f'adata_Protein_train.h5ad')

    adata_RNA_test.write_h5ad(save_path + f'adata_RNA_test.h5ad')
    adata_Protein_test.write_h5ad(save_path + f'adata_Protein_test.h5ad')

    print("Saved adata.")


def prepare_data_PBMC_full(adata_RNA, adata_Protein, train: bool = True, save_path: str = '', is_hvg_RNA: bool = True, is_hvg_protein: bool = False):
    print("Read PBMC data.")
    print(f"Train={train} RNA data shape {adata_RNA.shape}")
    print(f"Train={train} Protein data shape {adata_Protein.shape}")

    # Create PCA for benchmarking
    adata_merged_tmp = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged_tmp)

    adata_RNA, cells_subset = preprocess_rna(adata_RNA,min_features = 0, is_hvg=is_hvg_RNA, batch_key='batch')
    adata_Protein = adata_Protein[:, cells_subset]

    adata_RNA.obs['cell_type_l1'] = adata_RNA.obs['cell_type'].map(l2tol1)
    adata_Protein.obs['cell_type_l1'] = adata_Protein.obs['cell_type'].map(l2tol1)
    
    # Add PCA after preprocessing for benchmarking
    adata_RNA.write_h5ad(save_path + f'adata_RNA_full.h5ad')
    adata_Protein.write_h5ad(save_path + f'adata_Protein_full.h5ad')

    print("Saved adata.")

def read_data(data: str = "simulated", save_path: str = ""):
    if data == "simulated":
        path = './Multimodal_pretraining/data/multi_gene_l2.loom'
        adata_RNA = sc.read(path)

        path = './Multimodal_pretraining/data/multi_protein_l2.loom'
        adata_Protein = sc.read(path) #cell_type batch

        train_idx = (adata_RNA.obs["batch"] != "P2") & (adata_RNA.obs["batch"] != "P5") & (adata_RNA.obs["batch"] != "P8")
        test_idx = (train_idx != 1)

        # TODO Remove cell type B from reference
        RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_merged_test, adata_RNA_test  = prepare_data_PBMC_together(adata_RNA=adata_RNA, adata_Protein=adata_Protein, train=True, save_path=save_path, train_idx=train_idx, test_idx=test_idx)

    return RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_merged_test, adata_RNA_test

def main():
    # Read data
    save_path = './Multimodal_pretraining/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_merged_test, adata_RNA_test = read_data(data=data, save_path=save_path)


main()
