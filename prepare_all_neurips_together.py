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
    'Plasma cell IGKC+': 'B', #'Plasma cells',
    'Plasma cell IGKC-': 'B', #'Plasma cells',

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

l2tol1_multiome = {
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
        adata_cite,
        adata_multiome,
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

    sym_diff = list(set(adata_cite.var_names).symmetric_difference(set(adata_multiome.var_names)))
    adata_cite = adata_cite[:, [gene for gene in adata_cite.var_names
                      if str(gene) not in sym_diff]]
    adata_multiome = adata_multiome[:, [gene for gene in adata_multiome.var_names
                        if str(gene) not in sym_diff]]
    
    cells_subset1, _ = sc.pp.filter_cells(adata_cite, min_genes=min_features, inplace=False)
    adata_cite = adata_cite[cells_subset1, :]

    cells_subset2, _ = sc.pp.filter_cells(adata_multiome, min_genes=min_features, inplace=False)
    adata_multiome = adata_multiome[cells_subset2, :]
    
    adata = ad.concat([adata_cite, adata_multiome], axis=0)
    adata.obs["dataset"] = np.concatenate([np.zeros(shape=(adata_cite.shape[0])), np.ones(shape=(adata_multiome.shape[0]))], axis=0)
    print(adata)

    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    adata = adata[:, [gene for gene in adata.var_names
                      if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    
    if is_hvg == True:
        # sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key=batch_key, inplace=True, subset=True)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key='obs', inplace=True, subset=True)

    print('Processed dataset shape: {}'.format(adata.shape))

    adata_cite_new = adata[adata.obs["dataset"] == 0, :]
    adata_multiome_new = adata[adata.obs["dataset"] == 1, :]
    return adata_cite_new, cells_subset1, adata_multiome_new, cells_subset2

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

def prepare_data_neurips_together(adata_RNA_cite, adata_Protein_cite, 
                                  save_path_cite, train_idx_cite, 
                                  test_idx_cite,
                                  adata_RNA_multiome, adata_Protein_multiome, 
                                  save_path_multiome, train_idx_multiome, 
                                  test_idx_multiome):
    adata_RNA_cite, cells_subset_cite, adata_RNA_multiome, cells_subset_multiome = preprocess_rna(adata_RNA_cite, adata_RNA_multiome, min_features = 0, is_hvg=True, batch_key='batch')
    adata_Protein_cite = preprocess_protein(adata_Protein_cite[cells_subset_cite, :], min_features = 0, is_hvg=False, batch_key='batch')
    adata_Protein_multiome = preprocess_atac(adata_Protein_multiome[cells_subset_multiome, :], min_features = 0, is_hvg=True, batch_key='batch')
    
    adata_RNA_cite.obs['cell_type_l1'] = adata_RNA_cite.obs['cell_type'].map(l2tol1)
    adata_Protein_cite.obs['cell_type_l1'] = adata_Protein_cite.obs['cell_type'].map(l2tol1)
    adata_RNA_test_cite = adata_RNA_cite[test_idx_cite, :]
    adata_Protein_test_cite = adata_Protein_cite[test_idx_cite, :]
    adata_RNA_cite = adata_RNA_cite[train_idx_cite, :]
    adata_Protein_cite = adata_Protein_cite[train_idx_cite, :]

    adata_RNA_multiome.obs['cell_type_l1'] = adata_RNA_multiome.obs['cell_type'].map(l2tol1_multiome)
    adata_Protein_multiome.obs['cell_type_l1'] = adata_Protein_multiome.obs['cell_type'].map(l2tol1_multiome)
    adata_RNA_test_multiome = adata_RNA_multiome[test_idx_multiome, :]
    adata_Protein_test_multiome = adata_Protein_multiome[test_idx_multiome, :]
    adata_RNA_multiome = adata_RNA_multiome[train_idx_multiome, :]
    adata_Protein_multiome = adata_Protein_multiome[train_idx_multiome, :]

    # Write

    adata_RNA_cite.write_h5ad(save_path_cite + f'adata_cellbind_GEX_train.h5ad')
    adata_Protein_cite.write_h5ad(save_path_cite + f'adata_cellbind_ADT_train.h5ad')
    adata_RNA_test_cite.write_h5ad(save_path_cite + f'adata_cellbind_GEX_test.h5ad')
    adata_Protein_test_cite.write_h5ad(save_path_cite + f'adata_cellbind_ADT_test.h5ad')

    path_file_cite = 'tfrecord_train/'
    RNA_tf_path_cite = save_path_cite + path_file_cite + 'GEX_cellbind_tf/'
    Protein_tf_path_cite = save_path_cite + path_file_cite + 'ADT_cellbind_tf/'
    RNA_tf_path_cite = concerto_make_tfrecord(adata_RNA_cite,tf_path = RNA_tf_path_cite, batch_col_name = 'batch')
    Protein_tf_path_cite = concerto_make_tfrecord(adata_Protein_cite,tf_path = Protein_tf_path_cite, batch_col_name = 'batch')

    path_file_cite = 'tfrecord_test/'
    RNA_tf_path_test_cite = save_path_cite + path_file_cite + 'GEX_cellbind_tf/'
    Protein_tf_path_test_cite = save_path_cite + path_file_cite + 'ADT_cellbind_tf/'
    RNA_tf_path_test_cite = concerto_make_tfrecord(adata_RNA_test_cite,tf_path = RNA_tf_path_test_cite, batch_col_name = 'batch')
    Protein_tf_path_test_cite = concerto_make_tfrecord(adata_Protein_test_cite,tf_path = Protein_tf_path_test_cite, batch_col_name = 'batch')



    adata_RNA_multiome.write_h5ad(save_path_multiome + f'adata_cellbind_GEX_multiome_train.h5ad')
    adata_Protein_multiome.write_h5ad(save_path_multiome + f'adata_cellbind_ATAC_multiome_train.h5ad')

    adata_RNA_test_multiome.write_h5ad(save_path_multiome + f'adata_cellbind_GEX_multiome_test.h5ad')
    adata_Protein_test_multiome.write_h5ad(save_path_multiome + f'adata_cellbind_ATAC_multiome_test.h5ad')

    path_file_multiome = 'tfrecord_train/'
    RNA_tf_path_multiome = save_path_multiome + path_file_multiome + 'GEX_cellbind_multiome_tf/'
    Protein_tf_path_multiome = save_path_multiome + path_file_multiome + 'ATAC_cellbind_multiome_tf/'
    RNA_tf_path_multiome = concerto_make_tfrecord(adata_RNA_multiome,tf_path = RNA_tf_path_multiome, batch_col_name = 'batch')
    Protein_tf_path_multiome = concerto_make_tfrecord(adata_Protein_multiome,tf_path = Protein_tf_path_multiome, batch_col_name = 'batch')

    path_file_multiome = 'tfrecord_test/'
    RNA_tf_path_test_multiome = save_path_multiome + path_file_multiome + 'GEX_cellbind_multiome_tf/'
    Protein_tf_path_test_multiome = save_path_multiome + path_file_multiome + 'ATAC_cellbind_multiome_tf/'
    RNA_tf_path_test_multiome = concerto_make_tfrecord(adata_RNA_test_multiome,tf_path = RNA_tf_path_test_multiome, batch_col_name = 'batch')
    Protein_tf_path_test_multiome = concerto_make_tfrecord(adata_Protein_test_multiome,tf_path = Protein_tf_path_test_multiome, batch_col_name = 'batch')

    print("Saved adata and tf.")


def read_data(save_path_cite: str = "", save_path_multiome: str = ""):
    path_cite = 'Multimodal_pretraining/data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad'

    adata_merged_tmp_cite = sc.read_h5ad(path_cite)
    adata_RNA_cite = adata_merged_tmp_cite[:, 0:13953] # gex
    adata_RNA_cite.X = adata_RNA_cite.layers["counts"]
    adata_Protein_cite = adata_merged_tmp_cite[:, 13953:] # adt
    
    train_idx_cite = (adata_RNA_cite.obs["batch"] != "s4d1") & (adata_RNA_cite.obs["batch"] != "s4d8") & (adata_RNA_cite.obs["batch"] != "s4d9")
    test_idx_cite = (train_idx_cite != 1)



    path_multiome = 'Multimodal_pretraining/data/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad'

    adata_merged_tmp_multiome = sc.read_h5ad(path_multiome)
    adata_RNA_multiome = adata_merged_tmp_multiome[:, 0:13431] # gex
    adata_RNA_multiome.X = adata_RNA_multiome.layers["counts"]
    adata_Protein_multiome = adata_merged_tmp_multiome[:, 13431:] # atac
    
    train_idx_multiome = (adata_RNA_multiome.obs["batch"] != "s4d1") & (adata_RNA_multiome.obs["batch"] != "s4d8") & (adata_RNA_multiome.obs["batch"] != "s4d9")
    test_idx_multiome = (train_idx_multiome != 1)


    prepare_data_neurips_together(adata_RNA_cite=adata_RNA_cite, adata_Protein_cite=adata_Protein_cite, 
                                  save_path_cite=save_path_cite, train_idx_cite=train_idx_cite, 
                                  test_idx_cite=test_idx_cite,
                                  adata_RNA_multiome=adata_RNA_multiome, adata_Protein_multiome=adata_Protein_multiome, 
                                  save_path_multiome=save_path_multiome, train_idx_multiome=train_idx_multiome, 
                                  test_idx_multiome=test_idx_multiome)

def main():
    # Read data
    save_path = './Multimodal_pretraining/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    read_data(save_path=save_path)

main()
