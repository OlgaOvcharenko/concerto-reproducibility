import os
import sys

from matplotlib import pyplot as plt
import pandas as pd
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import issparse

from concerto_function5_3 import concerto_make_tfrecord
sys.path.append("../")
import numpy as np
import scanpy as sc
import anndata as ad
import episcanpy as epi

l2tol1 = {
    'CD8+ T': 'CD8 T',
    'CD8+ T naive': 'CD8 T',
    'CD4+ T activated': 'CD4 T',
    'CD4+ T naive': 'CD4 T',
    'CD14+ Mono': 'Monocytes',
    'CD16+ Mono': 'Monocytes',
    'NK': 'NK',
    'cDC2': 'DC',
    'pDC': 'DC',
    'B1 B': 'B',
    'Plasma cell': 'B', #'Plasma cells',
    'Transitional B': 'B',
    'Naive CD20+ B': 'B',
    'Erythroblast': 'Erythrocytes',
    'Proerythroblast': 'Erythrocytes',
    'Normoblast': 'Erythrocytes',
    'HSC': 'HSC', 
    'Lymph prog': 'Lymph prog',
    'G/M prog': 'G/M prog',
    'MK/E prog': 'MK/E prog',  
    'ID2-hi myeloid prog': 'other',
    'ILC': 'other',    
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

    cells_subset, _ = sc.pp.filter_cells(adata, min_genes=min_features, inplace=False)
    adata = adata[cells_subset, :]

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    
    if is_hvg == True:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key=batch_key, inplace=True, subset=True)

    print('Processed dataset shape: {}'.format(adata.shape))
    return adata, cells_subset

def preprocess_atac(
        adata,
        min_features: int = 600,
        min_cells: int = 5,
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
    #print(adata)
    #print(epi.pp.filter_cells(adata, min_features=min_features))
    #cells_subset, _ = epi.pp.filter_cells(adata, min_features=min_features, inplace=False)
    #adata = adata[cells_subset, :]

    epi.pp.filter_features(adata, min_cells=min_cells)

    # create a new AnnData containing only the most variable features
    nb_feature_selected = 10000
    adata.raw = adata
    adata = epi.pp.select_var_feature(adata,
                                    nb_features=nb_feature_selected,
                                    show=False,
                                    copy=True)
    
    epi.pp.normalize_total(adata)
    epi.pp.log1p(adata)
    
    # save the current version of the matrix (normalised) in a layer of the Anndata.
    adata.layers['normalised'] = adata.X.copy()
    print(adata)

    print('Processed dataset shape: {}'.format(adata.shape))
    return adata

def prepare_data_neurips_full(adata_RNA, adata_Protein, save_path: str = '', is_hvg_RNA: bool = True, is_hvg_protein: bool = False):
    print("Read PBMC data.")
    print(f"RNA data shape {adata_RNA.shape}")
    print(f"Protein data shape {adata_Protein.shape}")
    
    # Create PCA for benchmarking
    adata_merged_tmp = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged_tmp)

    adata_RNA, cells_subset = preprocess_rna(adata_RNA, min_features = 0, is_hvg=is_hvg_RNA, batch_key='batch')
    adata_Protein = preprocess_atac(adata_Protein[cells_subset, :], min_features = 0, is_hvg=is_hvg_RNA, batch_key='batch')

    #adata_RNA = adata_RNA[cells_subset, :]

    adata_RNA.obs['cell_type_l1'] = adata_RNA.obs['cell_type'].map(l2tol1)
    adata_Protein.obs['cell_type_l1'] = adata_Protein.obs['cell_type'].map(l2tol1)
    
    # print(adata_RNA.obs['cell_type_l1'].value_counts())
    # adata_RNA.obsm["X_umap"] = adata_RNA.obsm["GEX_X_umap"] 
    # sc.pl.umap(adata_RNA, color=["cell_type_l1"], legend_fontweight='light') 
    # plt.savefig("tmp.png")

    # ix = (adata_RNA.obs['cell_type_l1'] != 'other') & (adata_RNA.obs['cell_type_l1'] != 'other T')
    # adata_RNA = adata_RNA[ix, :]
    # adata_Protein = adata_Protein[ix, :]

    # Add PCA after preprocessing for benchmarking
    adata_RNA.write_h5ad(save_path + f'adata_neurips_GEX_multiome_full.h5ad')
    adata_Protein.write_h5ad(save_path + f'adata_neurips_ATAC_multiome_full.h5ad')

    path_file = 'tfrecord_full/'
    RNA_tf_path = save_path + path_file + 'GEX_multiome_tf/'
    Protein_tf_path = save_path + path_file + 'ATAC_multiome_tf/'
    RNA_tf_path = concerto_make_tfrecord(adata_RNA,tf_path = RNA_tf_path, batch_col_name = 'batch')
    Protein_tf_path = concerto_make_tfrecord(adata_Protein,tf_path = Protein_tf_path, batch_col_name = 'batch')

    print("Saved adata and tf.")

def prepare_data_neurips_together(train_idx, test_idx, adata_RNA, adata_Protein, save_path: str = '', is_hvg_RNA: bool = True, is_hvg_protein: bool = False):
    print("Read PBMC data.")
    print(f"RNA data shape {adata_RNA.shape}")
    print(f"Protein data shape {adata_Protein.shape}")

    adata_merged_tmp = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged_tmp)

    adata_RNA, cells_subset = preprocess_rna(adata_RNA, min_features = 0, is_hvg=is_hvg_RNA, batch_key='batch')
    adata_Protein  = preprocess_atac(adata_Protein[cells_subset, :], min_features = 0, is_hvg=is_hvg_RNA, batch_key='batch')

    #adata_RNA = adata_RNA[cells_subset, :]

    adata_RNA.obs['cell_type_l1'] = adata_RNA.obs['cell_type'].map(l2tol1)
    adata_Protein.obs['cell_type_l1'] = adata_Protein.obs['cell_type'].map(l2tol1)
    
    adata_RNA_test = adata_RNA[test_idx, :]
    adata_Protein_test = adata_Protein[test_idx, :]

    adata_RNA = adata_RNA[train_idx, :]
    adata_Protein = adata_Protein[train_idx, :]

    # ix = (adata_RNA.obs['cell_type_l1'] != 'other') & (adata_RNA.obs['cell_type_l1'] != 'other T')
    # adata_RNA = adata_RNA[ix, :]
    # adata_Protein = adata_Protein[ix, :]

    # ix = (adata_RNA_test.obs['cell_type_l1'] != 'other') & (adata_RNA_test.obs['cell_type_l1'] != 'other T')
    # adata_RNA_test = adata_RNA_test[ix, :]
    # adata_Protein_test = adata_Protein_test[ix, :]

    adata_RNA.write_h5ad(save_path + f'adata_GEX_multiome_train.h5ad')
    adata_Protein.write_h5ad(save_path + f'adata_ATAC_multiome_train.h5ad')

    adata_RNA_test.write_h5ad(save_path + f'adata_GEX_multiome_test.h5ad')
    adata_Protein_test.write_h5ad(save_path + f'adata_ATAC_multiome_test.h5ad')

    path_file = 'tfrecord_train/'
    RNA_tf_path = save_path + path_file + 'GEX_multiome_tf/'
    Protein_tf_path = save_path + path_file + 'ATAC_multiome_tf/'
    RNA_tf_path = concerto_make_tfrecord(adata_RNA,tf_path = RNA_tf_path, batch_col_name = 'batch')
    Protein_tf_path = concerto_make_tfrecord(adata_Protein,tf_path = Protein_tf_path, batch_col_name = 'batch')

    path_file = 'tfrecord_test/'
    RNA_tf_path_test = save_path + path_file + 'GEX_multiome_tf/'
    Protein_tf_path_test = save_path + path_file + 'ATAC_multiome_tf/'
    RNA_tf_path_test = concerto_make_tfrecord(adata_RNA_test,tf_path = RNA_tf_path_test, batch_col_name = 'batch')
    Protein_tf_path_test = concerto_make_tfrecord(adata_Protein_test,tf_path = Protein_tf_path_test, batch_col_name = 'batch')

    print("Saved adata and tf.")


def read_data(save_path: str = ""):
    path = 'Multimodal_pretraining/data/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad'

    adata_merged_tmp = sc.read_h5ad(path)
    adata_RNA = adata_merged_tmp[:, 0:13431] # gex
    adata_RNA.X = adata_RNA.layers["counts"]
    adata_Protein = adata_merged_tmp[:, 13431:] # atac
    
    train_idx = (adata_RNA.obs["batch"] != "s4d1") & (adata_RNA.obs["batch"] != "s4d8") & (adata_RNA.obs["batch"] != "s4d9")
    test_idx = (train_idx != 1)

    prepare_data_neurips_together(adata_RNA=adata_RNA, adata_Protein=adata_Protein, save_path=save_path, train_idx=train_idx, test_idx=test_idx)
    prepare_data_neurips_full(adata_RNA=adata_RNA, adata_Protein=adata_Protein, save_path=save_path)

def main():
    # Read data
    save_path = './Multimodal_pretraining/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    read_data(save_path=save_path)

main()
