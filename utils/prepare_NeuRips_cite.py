import os
import sys

from matplotlib import pyplot as plt
import pandas as pd
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import issparse
sys.path.append("../")
import numpy as np
import scanpy as sc
import anndata as ad

l2tol1 = {
    'CD8 Naive': 'CD8 T',
    'CD8 Proliferating': 'CD8 T',
    'CD8 TCM': 'CD8 T',
    'CD8 TEM': 'CD8 T',
    'CD8+ T': 'CD8 T',
    'CD8+ T CD57+ CD45RO+': 'CD8 T',
    'CD8+ T naive': 'CD8 T',
    'CD4 CTL': 'CD4 T',
    'CD4 Naive': 'CD4 T',
    'CD4 Proliferating': 'CD4 T',
    'CD4 TCM': 'CD4 T',
    'CD4 TEM': 'CD4 T',
    'CD4+ T activated': 'CD4 T',
    'CD4+ T naive': 'CD4 T',
    'CD8+ T CD57+ CD45RA+': 'CD8 T',
    'CD8+ T TIGIT+ CD45RO+': 'CD8 T',
    'CD4+ T activated integrinB7+': 'CD4 T',   
    'CD8+ T TIGIT+ CD45RA+': 'CD8 T', 
    'CD8+ T CD49f+': 'CD8 T', 
    'CD8+ T CD69+ CD45RA+': 'CD8 T',
    'CD8+ T CD69+ CD45RO+': 'CD8 T',
    'Naive CD20+ B IGKC+': 'B',
    'Naive CD20+ B IGKC-': 'B',
    'CD14+ Mono': 'Monocytes',
    'CD16+ Mono': 'Monocytes',
    'Treg': 'CD4 T',
    'NK': 'NK',
    'NK Proliferating': 'NK',
    'NK_CD56bright': 'NK',
    'NK CD158e1+': 'NK',
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
    'B1 B IGKC+': 'B',
    'B1 B IGKC-': 'B',
    'Plasmablast': 'B',
    'Plasma cell': 'B',
    'Transitional B': 'B',
    'Naive CD20+ B': 'B',
    'Eryth': 'other',
    'HSPC': 'other',
    'Platelet': 'other',
    'Doublet': 'other',
    'Erythroblast': 'Erythrocytes',
    'Proerythroblast': 'Erythrocytes',
    'Normoblast': 'Erythrocytes',
    'HSC': 'HSC', 

    'Lymph prog': 'Lymph prog',
    'G/M prog': 'G/M prog',
    'MK/E prog': 'MK/E prog',  

    'Reticulocyte': 'Erythrocytes',
    'Plasma cell IGKC+': 'Plasma cells',
    'Plasma cell IGKC-': 'Plasma cells',

    # FIXME
    'CD4+ T CD314+ CD45RA+': 'CD4 T',
    'CD8+ T naive CD127+ CD26- CD101-': 'CD8 T', 

    'gdT': 'other T',
    'gdT CD158b+':'other T',
    'T reg': 'other T',
    'gdT TCRVD2+': 'other T',                     
    'MAIT': 'other T', 
    'dnT': 'other',
    'ID2-hi myeloid prog': 'other',
    'ILC1': 'other',      
    'ILC': 'other',    
    'Plasmablast IGKC+': 'other',
    'Plasmablast IGKC-': 'other',     
    'T prog cycling': 'other',
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

    # print(cells_subset)
    # print(adata)

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    
    if is_hvg == True:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key=batch_key, inplace=True, subset=True)

    print('Processed dataset shape: {}'.format(adata.shape))
    return adata, cells_subset

def preprocess_protein(
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
    
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

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
    adata_Protein = preprocess_protein(adata_Protein[cells_subset, :], min_features = 0, is_hvg=is_hvg_RNA, batch_key='batch')

    adata_RNA.obs['cell_type_l1'] = adata_RNA.obs['cell_type'].map(l2tol1)
    adata_Protein.obs['cell_type_l1'] = adata_Protein.obs['cell_type'].map(l2tol1)
    
    print(adata_RNA.obs['cell_type_l1'].value_counts())
    adata_RNA.obsm["X_umap"] = adata_RNA.obsm["GEX_X_umap"] 
    sc.pl.umap(adata_RNA, color=["cell_type_l1"], legend_fontweight='light') 
    plt.savefig("tmp.png")

    ix = (adata_RNA.obs['cell_type_l1'] != 'other') & (adata_RNA.obs['cell_type_l1'] != 'other T')
    adata_RNA = adata_RNA[ix, :]
    adata_Protein = adata_Protein[ix, :]

    # Add PCA after preprocessing for benchmarking
    adata_RNA.write_h5ad(save_path + f'data/data/adata_neurips_GEX_full.h5ad')
    adata_Protein.write_h5ad(save_path + f'data/data/adata_neurips_ADT_full.h5ad')

    print("Saved adata.")

def read_data(save_path: str = ""):
    path = 'Multimodal_pretraining/data/data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad'

    adata_merged_tmp = sc.read_h5ad(path)
    adata_RNA = adata_merged_tmp[:, 0:13953] # gex
    adata_RNA.X = adata_RNA.layers["counts"]
    adata_Protein = adata_merged_tmp[:, 13953:] # adt
    
    prepare_data_neurips_full(adata_RNA=adata_RNA, adata_Protein=adata_Protein, save_path=save_path)

def main():
    # Read data
    save_path = './Multimodal_pretraining/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    read_data(save_path=save_path)

main()
