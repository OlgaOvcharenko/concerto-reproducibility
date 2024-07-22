import os
import sys

import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
sys.path.append("../")
from concerto_function5_3 import *
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
# Inital setting for plot size
from matplotlib import gridspec, rcParams
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, silhouette_samples
import tensorflow as tf
from sklearn.metrics import confusion_matrix
# import scvelo as scv
from sklearn.decomposition import KernelPCA

import time

def get_args():
    parser = argparse.ArgumentParser(description='CONCERTO Batch Correction.')

    parser.add_argument('--data', type=str, required=True,
                        help='Dataset (Simulated/Not)')

    parser.add_argument('--epoch', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--lr', type= float, required=True,
                        help='learning rate')
    parser.add_argument('--batch_size', type= int, required=True,
                        help='batch size')
    parser.add_argument('--drop_rate', type= float, required=True,
                        help='dropout rate')
    parser.add_argument('--heads', type= int, required=True,
                        help='heads for NNs')
    parser.add_argument('--attention_t', type= int, required=True,
                        help='to use attention with teacher')
    parser.add_argument('--attention_s', type= int, required=True,
                        help='to use attention with student')
    parser.add_argument('--train', type= int, required=True,
                        help='to train or just inference')
    parser.add_argument('--test', type= int, required=True,
                        help='inference')
    parser.add_argument('--model_type', type= int, required=True,
                        help='1 for simple TT, else 4 together')
    parser.add_argument('--combine_omics', type= int, required=True,
                        help='0/1')

    args = parser.parse_args()
    return args

def prepare_data_PBMC(adata_RNA, adata_Protein, train: bool = True, save_path: str = ''):
    print("Read PBMC data.")
    print(f"Train={train} RNA data shape {adata_RNA.shape}")
    print(f"Train={train} Protein data shape {adata_Protein.shape}")

    # Create PCA for benchmarking
    adata_merged_tmp = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged_tmp)

    adata_RNA = preprocessing_changed_rna(adata_RNA,min_features = 0, is_hvg=True, batch_key='batch')
    adata_Protein = preprocessing_changed_rna(adata_Protein,min_features = 0, is_hvg=True, batch_key='batch')
    
    # Add PCA after preprocessing for benchmarking
    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]
    adata_merged.obsm["Unintegrated"] = adata_merged_tmp.obsm["X_pca"]
    del adata_merged_tmp
    
    print("Preprocessed data.")

    print(f"RNA data shape {adata_RNA.shape}")
    print(f"Protein data shape {adata_Protein.shape}")
    print(f"RNA data: \n {adata_RNA}")
    print(f"Protein data: \n {adata_Protein}")

    adata_RNA.write_h5ad(save_path + f'adata_RNA_{"train" if train else "test"}.h5ad')
    adata_Protein.write_h5ad(save_path + f'adata_Protein_{"train" if train else "test"}.h5ad')

    print("Saved adata.")

    path_file = 'tfrecord_train/' if train else 'tfrecord_test/'
    RNA_tf_path = save_path + path_file + 'RNA_tf/'
    Protein_tf_path = save_path + path_file + 'Protein_tf/'

    RNA_tf_path = concerto_make_tfrecord(adata_RNA,tf_path = RNA_tf_path, batch_col_name = 'batch')
    Protein_tf_path = concerto_make_tfrecord(adata_Protein,tf_path = Protein_tf_path, batch_col_name = 'batch')

    print("Made tf records.")

    return RNA_tf_path, Protein_tf_path, adata_merged


def prepare_data_neurips(adata_merged_tmp, adata_RNA, adata_Protein, train: bool = True, save_path: str = ''):
    print("Read human data")
    print(f"Train={train} gex data shape {adata_RNA.shape}")
    print(f"Train={train} atac data shape {adata_Protein.shape}")

    # Create PCA for benchmarking
    sc.tl.pca(adata_merged_tmp)

    # FIXME why 20K
    adata_RNA = preprocessing_changed_rna(adata_RNA, min_features=0, is_hvg=True, batch_key='batch')
    adata_Protein = preprocessing_changed_rna(adata_Protein, min_features=0, is_hvg=True, batch_key='batch')
    
    # Add PCA after preprocessing for benchmarking
    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]
    adata_merged_tmp.obsm["Unintegrated"] = adata_merged_tmp.obsm["X_pca"]

    del adata_merged_tmp
    
    print("Preprocessed data.")

    print(f"GEX data shape {adata_RNA.shape}")
    print(f"ATAC data shape {adata_Protein.shape}")
    print(f"GEX data: \n {adata_RNA}")
    print(f"ATAC data: \n {adata_Protein}")

    adata_RNA.write_h5ad(save_path + f'adata_gex_{"train" if train else "test"}.h5ad')
    adata_Protein.write_h5ad(save_path + f'adata_atac_{"train" if train else "test"}.h5ad')
    print("Saved adata.")

    path_file = 'tfrecord_train/' if train else 'tfrecord_test/'
    RNA_tf_path = save_path + path_file + 'gex_tf/'
    Protein_tf_path = save_path + path_file + 'atac_tf/'

    RNA_tf_path = concerto_make_tfrecord(adata_RNA, tf_path = RNA_tf_path, batch_col_name = 'batch')
    Protein_tf_path = concerto_make_tfrecord(adata_Protein, tf_path = Protein_tf_path, batch_col_name = 'batch')
    print("Made tf record.")

    return RNA_tf_path, Protein_tf_path, adata_merged
    

def read_data(data: str = "simulated", save_path: str = ""):
    if data == "simulated":
        path = './Multimodal_pretraining/data/multi_gene_l2.loom'
        adata_RNA = sc.read(path)

        path = './Multimodal_pretraining/data/multi_protein_l2.loom'
        adata_Protein = sc.read(path) #cell_type batch

        # train_idx, test_idx = train_test_split(
        #     adata_RNA.obs_names.values,
        #     test_size=0.5,
        #     stratify=adata_RNA.obs["batch_size"],
        #     shuffle=False,
        #     random_state=42,
        # )

        train_idx = (adata_RNA.obs["batch"] != "P6") & (adata_RNA.obs["batch"] != "P7") & (adata_RNA.obs["batch"] != "P8")
        test_idx = (train_idx != 1)

        # ['ASDC' 'B intermediate' 'B memory' 'B naive' 'CD14 Mono' 'CD16 Mono'
        # 'CD4 CTL' 'CD4 Naive' 'CD4 Proliferating' 'CD4 TCM' 'CD4 TEM' 'CD8 Naive'
        # 'CD8 Proliferating' 'CD8 TCM' 'CD8 TEM' 'Doublet' 'Eryth' 'HSPC' 'ILC'
        # 'MAIT' 'NK' 'NK Proliferating' 'NK_CD56bright' 'Plasmablast' 'Platelet'
        # 'Treg' 'cDC1' 'cDC2' 'dnT' 'gdT' 'pDC']

        adata_RNA_test = adata_RNA[test_idx, :]
        adata_Protein_test = adata_Protein[test_idx, :]

        adata_RNA = adata_RNA[train_idx, :]
        adata_Protein = adata_Protein[train_idx, :]

        # TODO Remove cell type B from reference
        # cell_ix = (adata_RNA.obs["cell_type"] != "B intermediate") & (adata_RNA.obs["cell_type"] != "B memory") & (adata_RNA.obs["cell_type"] != "B naive") & (adata_RNA.obs["cell_type"] != "Plasmablast")
        # adata_RNA = adata_RNA[cell_ix, :]
        # adata_Protein = adata_Protein[cell_ix, :]

        RNA_tf_path, Protein_tf_path, adata_merged = prepare_data_PBMC(adata_RNA=adata_RNA, adata_Protein=adata_Protein, train=True, save_path=save_path)
        RNA_tf_path_test, Protein_tf_path_test, adata_merged_test = prepare_data_PBMC(adata_RNA=adata_RNA_test, adata_Protein=adata_Protein_test, train=False, save_path=save_path)

    else:
        adata_merged_tmp = sc.read_h5ad("./Multimodal_pretraining/data/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad")
        adata_RNA = adata_merged_tmp[:, 0:13431] # adata_gex
        adata_Protein = adata_merged_tmp[:, 13431:] # adata_atac

        # train_idx, test_idx = train_test_split(
        #     adata_RNA.obs_names.values,
        #     test_size=0.5,
        #     stratify=adata_RNA.obs["batch_size"],
        #     shuffle=False,
        #     random_state=42,
        # )

        # ['s1d1' 's1d2' 's1d3' 's2d1' 's2d4' 's2d5' 's3d10' 's3d3' 's3d6' 's3d7' 's4d1' 's4d8' 's4d9']

        train_idx = (adata_RNA.obs["batch"] != "s4d1") & (adata_RNA.obs["batch"] != "s4d8") & (adata_RNA.obs["batch"] != "s4d9")
        test_idx = (train_idx != 1)

        # ['B1 B' 'CD14+ Mono' 'CD16+ Mono' 'CD4+ T activated' 'CD4+ T naive'
        # 'CD8+ T' 'CD8+ T naive' 'Erythroblast' 'G/M prog' 'HSC'
        # 'ID2-hi myeloid prog' 'ILC' 'Lymph prog' 'MK/E prog' 'NK' 'Naive CD20+ B'
        # 'Normoblast' 'Plasma cell' 'Proerythroblast' 'Transitional B' 'cDC2'
        # 'pDC']

        adata_RNA_test = adata_RNA[test_idx, :]
        adata_Protein_test = adata_Protein[test_idx, :]
        adata_merged_tmp_test = adata_merged_tmp[test_idx, :]

        adata_RNA = adata_RNA[train_idx, :]
        adata_Protein = adata_Protein[train_idx, :]
        adata_merged_tmp = adata_merged_tmp[train_idx, :]

        RNA_tf_path, Protein_tf_path, adata_merged = prepare_data_neurips(adata_merged_tmp=adata_merged_tmp, adata_RNA=adata_RNA, adata_Protein=adata_Protein, train=True, save_path=save_path)
        RNA_tf_path_test, Protein_tf_path_test, adata_merged_test = prepare_data_neurips(adata_merged_tmp=adata_merged_tmp_test, adata_RNA=adata_RNA_test, adata_Protein=adata_Protein_test, train=False, save_path=save_path)

    return RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_merged_test, adata_RNA_test

def train_concerto(weight_path: str, RNA_tf_path: str, Protein_tf_path: str, data: str, 
                   attention_t: bool, attention_s: bool,
                   batch_size:int, epoch: int, lr: float, drop_rate: float, 
                   heads: int, combine_omics: int, model_type: int):
    # Train
    if attention_t == True and attention_s == False:
        concerto_train_multimodal(['RNA','Protein'] if data == 'simulated' else ['GEX', 'ATAC'], 
                                RNA_tf_path, 
                                Protein_tf_path, 
                                weight_path, 
                                super_parameters={
                                    'data': data,
                                    'batch_size': batch_size, 
                                    'epoch_pretrain': epoch, 'lr': lr, 
                                    'drop_rate': drop_rate, 
                                    'attention_t': attention_t, 
                                    'attention_s': attention_s, 
                                    'heads': heads,
                                    'combine_omics': combine_omics,
                                    'model_type': model_type
                                    })
    elif attention_t == True and attention_s == True:
        concerto_train_multimodal_tt(['RNA','Protein'] if data == 'simulated' else ['GEX', 'ATAC'],
                                RNA_tf_path, 
                                Protein_tf_path, 
                                weight_path, 
                                super_parameters={
                                    'data': data,
                                    'batch_size': batch_size, 
                                    'epoch_pretrain': epoch, 'lr': lr, 
                                    'drop_rate': drop_rate, 
                                    'attention_t': attention_t, 
                                    'attention_s': attention_s, 
                                    'heads': heads,
                                    'combine_omics': combine_omics
                                    })
    elif attention_t == False and attention_s == False:
        concerto_train_multimodal_ss(['RNA','Protein'] if data == 'simulated' else ['GEX', 'ATAC'],
                                RNA_tf_path,
                                Protein_tf_path,
                                weight_path, 
                                super_parameters={
                                    'data': data,
                                    'batch_size': batch_size, 
                                    'epoch_pretrain': epoch, 'lr': lr, 
                                    'drop_rate': drop_rate, 
                                    'attention_t': attention_t, 
                                    'attention_s': attention_s, 
                                    'heads': heads,
                                    'combine_omics': False
                                    })

    print("Trained.")


def test_concerto(adata_merged, adata_RNA, weight_path: str, RNA_tf_path_test: str, Protein_tf_path_test: str, data: str, 
                   attention_t: bool, attention_s: bool,
                   batch_size:int, epoch: int, lr: float, drop_rate: float, 
                   heads: int, combine_omics: int, model_type: int, 
                   save_path: str, train: bool = False, adata_merged_train = None):
    ep_vals = []
    i = 4
    while i < epoch:
        ep_vals.append(i)
        i = i * 2
    ep_vals.append(epoch)

    adata_merged.obs = adata_RNA.obs

    print("Merged adata")
    print(adata_merged)

    # Test
    diverse_tests_names = []

    only_RNAs = [True, False] if combine_omics == 1 else [False]
    for only_RNA in only_RNAs:
        for dr in [drop_rate]:
            for nn in ["encoder"]:
                for e in ep_vals: 
                    saved_weight_path = f'./Multimodal_pretraining/weight/multi_weight_{nn}_{data}_{batch_size}_model_{combine_omics}_{model_type}_epoch_{e}_{lr}_{drop_rate}_{attention_t}_{attention_s}_{heads}.h5'

                    if (nn == "decoder" and attention_s == False) or (nn == "encoder" and attention_t == False):
                        embedding, batch, RNA_id, attention_weight =  concerto_test_multimodal_decoder(
                        ['RNA','Protein'] if data == 'simulated' else ['GEX', 'ATAC'],
                        weight_path, 
                        RNA_tf_path_test,
                        Protein_tf_path_test,
                        n_cells_for_sample=None,
                        super_parameters={
                            'batch_size': batch_size, 
                            'epoch': e, 'lr': lr, 
                            'drop_rate': dr, 
                            'attention_t': attention_t, 
                            'attention_s': attention_s, 
                            'heads': heads,
                            'combine_omics': combine_omics
                        }, 
                        saved_weight_path = saved_weight_path)
                    else:
                        embedding, batch, RNA_id, attention_weight =  concerto_test_multimodal(
                            ['RNA','Protein'] if data == 'simulated' else ['GEX', 'ATAC'],
                            weight_path, 
                            RNA_tf_path_test,
                            Protein_tf_path_test,
                            n_cells_for_sample=None,
                            super_parameters={
                                'batch_size': batch_size, 
                                'epoch_pretrain': e, 'lr': lr, 
                                'drop_rate': dr, 
                                'attention_t': attention_t, 
                                'attention_s': attention_s, 
                                'heads': heads,
                                'combine_omics': combine_omics,
                                'model_type': model_type
                            }, 
                            saved_weight_path = saved_weight_path,
                            only_RNA=only_RNA)

                    print("Tested.")
                    
                    if data == "simulated":
                        adata_RNA = sc.read(save_path + f'adata_RNA_{"train" if train else "test"}.h5ad')
                    else:
                        adata_RNA = sc.read(save_path + f'adata_gex_{"train" if train else "test"}.h5ad')
                    
                    adata_RNA_1 = adata_RNA[RNA_id]
                    adata_RNA_1.obsm['X_embedding'] = embedding

                    print(f"Shape of the embedding {embedding.shape}.")
                    
                    adata_merged = adata_merged[RNA_id]

                    adata_merged.obsm[f'{"train" if train else "test"}_{e}_{nn}_{dr}'] = embedding
                    diverse_tests_names.append(f"{train}_{e}_{nn}_{dr}")

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
                    adata_RNA_1.obs['cell_type_l1'] = adata_RNA_1.obs['cell_type'].map(l2tol1)
                    adata_merged.obs['cell_type_l1'] = adata_RNA_1.obs['cell_type'].map(l2tol1)
                    # else:
                    #     adata_RNA_1.obs['cell_type_l1'] = adata_RNA_1.obs['cell_type']
                    #     adata_merged.obs['cell_type_l1'] = adata_RNA_1.obs['cell_type']
                    print(adata_RNA_1)

                    sc.pp.neighbors(adata_RNA_1, use_rep="X_embedding", metric="cosine")
                    labels = adata_RNA_1.obs['cell_type_l1'].tolist()
                    for res in [0.05,0.1,0.15,0.2,0.25,0.3]:
                        sc.tl.leiden(adata_RNA_1, resolution=res)
                        target_preds = adata_RNA_1.obs['leiden'].tolist()
                        nmi = np.round(normalized_mutual_info_score(labels, target_preds), 5)
                        ari = np.round(adjusted_rand_score(labels, target_preds), 5)    
                        n_cluster = len(list(set(target_preds)))
                        print('leiden(res=%f): ari = %.5f , nmi = %.5f, n_cluster = %d' % (res, ari, nmi, n_cluster), '.')

                    if not train:
                        print("Predict")
                        adata_RNA_1.obs[f'pred_cell_type_{e}_{nn}_{dr}'] = query_to_reference(X_train=adata_merged_train.obsm[f'train_{e}_{nn}_{dr}'], y_train=adata_merged_train.obs["cell_type_l1"], X_test=adata_merged.obsm[f'test_{e}_{nn}_{dr}'], y_test=adata_merged.obs["cell_type_l1"], ).set_index(adata_RNA_1.obs_names)["val_ct"]
                        # print(adata_RNA_1.obs[f'pred_cell_type_{e}_{nn}_{dr}'])

                    # sc.pp.neighbors(adata_RNA_1, use_rep='X_embedding', metric='cosine')
                    sc.tl.leiden(adata_RNA_1, resolution=0.2)
                    sc.tl.umap(adata_RNA_1, min_dist=0.1)
                    adata_merged.obsm[f'{"train" if train else "test"}_umap_{e}_{nn}_{dr}'] = adata_RNA_1.obsm["X_umap"]
                    adata_merged.obs[f'{"train" if train else "test"}_leiden_{e}_{nn}_{dr}'] = adata_RNA_1.obs["leiden"]
                    sc.set_figure_params(dpi=150)

                    if not train:
                        color=['cell_type_l1', f'pred_cell_type_{e}_{nn}_{dr}', 'leiden', 'batch']
                        # color=['cell_type_l1', 'leiden', 'batch']
                    else:
                        color=['cell_type_l1', 'leiden', 'batch']
                    sc.pl.umap(adata_RNA_1, color=color, legend_fontsize ='xx-small', size=5, legend_fontweight='light', edges=True)
                    plt.savefig(f'./Multimodal_pretraining/plots/{data}/{data}_{"train" if train else "test"}_{combine_omics}_oRNA{only_RNA}_mt_{model_type}_bs_{batch_size}_{nn}_{e}_{lr}_{drop_rate}_{dr}_{attention_s}_{attention_t}_{heads}.png')
                    
                    # scv.pl.velocity_embedding(f'./Multimodal_pretraining/plots/{data}/{data}_mt_{model_type}_bs_{batch_size}_{nn}_{e}_{lr}_{drop_rate}_{dr}_{attention_s}_{attention_t}_{heads}.png', basis="umap")

    return adata_merged

def save_merged_adata(adata_merged, filename):
    adata_merged.write(filename)

    print(adata_merged)
    print(f"Saved adata all at {filename}")

def query_to_reference(X_train, X_test, y_train, y_test):
    # Prepare
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    transformer = KernelPCA(n_components=20, kernel='linear')
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    y_train = pd.DataFrame(y_train.to_list(), columns=["ct"])
    y_train.fillna(-1, inplace=True)

    y_test_original = y_test.to_list()

    y_test = pd.DataFrame(y_test.to_list(), columns=["ct"])
    y_test.fillna(-1, inplace=True)
    

    # Encode
    label_types = dict()
    i = 0
    for i, lbl in enumerate(y_train["ct"].unique()):
        label_types[lbl] = i
        y_train["ct"][y_train["ct"]==lbl] = i
        y_test["ct"][y_test["ct"]==lbl] = i

    i = len(y_train["ct"].unique())
    clusters_test_ix = np.ones((y_test.shape[0],), dtype=int)
    for lbl in y_test["ct"].unique():
        if (lbl not in list(label_types.keys())) and (lbl not in list(label_types.values())):
            clusters_test_ix[y_test["ct"]==lbl] = 0 # FIXME
            y_test["ct"][y_test["ct"]==lbl] = i
            label_types[lbl] = i
            i += 1

            print("Visited")

    y_train['ct'] = y_train['ct'].astype('int')
    y_test['ct'] = y_test['ct'].astype('int')

    # Fit
    neigh = KNeighborsClassifier(n_neighbors=100, metric='cosine')
    neigh.fit(X_train, y_train["ct"].to_numpy())

    # # Leiden
    # adata_new = ad.AnnData(np.append(X_train, X_test, axis=0))
    # sc.pp.neighbors(adata_new, metric="cosine", use_rep="X")
    # sc.tl.leiden(adata_new, resolution=0.2)

    # # FIXME Filter clusters
    # clusters_train = np.array(adata_new.obs["leiden"])[0:X_train.shape[0]]
    # clusters_test = np.array(adata_new.obs["leiden"])[X_train.shape[0]:(X_train.shape[0]+X_test.shape[0])]

    # clusters_test_ix = np.ones(clusters_test.shape, dtype=int)
    # print(np.unique(clusters_test).tolist())
    # print(np.unique(clusters_train).tolist())
    # for cl in set(np.unique(clusters_test).tolist()).difference(np.unique(clusters_train).tolist()):
    #     clusters_test_ix[clusters_test == cl] = 0

    clusters_test_ix = np.array(clusters_test_ix, dtype=bool)

    y_predicted = np.full((y_test.shape[0],), -1, dtype=int)
    y_predicted[clusters_test_ix] = neigh.predict(X_test[clusters_test_ix,:])
    print(y_predicted[clusters_test_ix].shape)

    print(f"Accuracy known: {accuracy_score(y_test['ct'][clusters_test_ix], y_predicted[clusters_test_ix])}")
    print(f"Accuracy known (my): {sum(y_test.loc[clusters_test_ix, 'ct'].to_numpy() == y_predicted[clusters_test_ix]) / y_test['ct'][clusters_test_ix].shape[0]}")

    y_predicted["ct"][clusters_test_ix != 1] == -2
    print(f"Accuracy all: {accuracy_score(y_test, y_predicted)}")
    print(label_types)
    print(list(label_types.values()))
    print(confusion_matrix(y_test['ct'][clusters_test_ix], y_predicted[clusters_test_ix], labels=list(label_types.values())))

    # print(y_test.value_counts())
    # print(np.unique(y_predicted, return_counts=True))
    
    y_predicted = pd.DataFrame(data=y_predicted, columns=["ct"])

    y_predicted['val_ct'] = pd.Series(dtype='int')
    for lbl, num_lbl in zip(list(label_types.keys()), list(label_types.values())):
        y_predicted["val_ct"][y_predicted["ct"]==num_lbl] = lbl
    
    y_predicted[y_predicted["ct"]==-2] == "new cell type"
    
    return y_predicted

def main():
    # Parse args
    args = get_args()
    data = args.data
    epoch = args.epoch
    lr = args.lr
    batch_size= args.batch_size
    drop_rate= args.drop_rate
    attention_t = True if args.attention_t == 1 else False
    attention_s = True if args.attention_s == 1 else False 
    heads = args.heads
    train = args.train 
    model_type = args.model_type
    test = args.test
    combine_omics = args.combine_omics

    print(f"Multimodal correction: epoch {epoch}, model type {model_type}, lr {lr}, batch_size {batch_size}, drop_rate {drop_rate}, attention_t {attention_t}, attention_s {attention_s}, heads {heads}.")

    # Check num GPUs
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(f"\nAvailable GPUs: {gpus}\n")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    
    # Read data
    save_path = './Multimodal_pretraining/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_merged_test, adata_RNA_test = read_data(data=data, save_path=save_path)

    # Train
    weight_path = save_path + 'weight/'
    if train:
        train_concerto(weight_path=weight_path, RNA_tf_path=RNA_tf_path, Protein_tf_path=Protein_tf_path, data=data, 
                   attention_t=attention_t, attention_s=attention_s, 
                   batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
                   heads=heads, combine_omics=combine_omics, model_type=model_type)
    print("Trained.")

    if test:
        # Test on train data
        adata_merged = test_concerto(weight_path=weight_path, RNA_tf_path_test=RNA_tf_path, Protein_tf_path_test=Protein_tf_path, data=data, 
                   attention_t=attention_t, attention_s=attention_s,
                   batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
                   heads=heads, combine_omics=combine_omics, model_type=model_type, 
                   save_path=save_path, train=True, adata_merged=adata_merged, adata_RNA=adata_RNA)

        # Test on test data
        adata_merged_test = test_concerto(weight_path=weight_path, RNA_tf_path_test=RNA_tf_path_test, Protein_tf_path_test=Protein_tf_path_test, data=data, 
                   attention_t=attention_t, attention_s=attention_s,
                   batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
                   heads=heads, combine_omics=combine_omics, model_type=model_type, 
                   save_path=save_path, train=False, adata_merged=adata_merged_test, adata_RNA=adata_RNA_test, adata_merged_train=adata_merged)
    
    filename = f'./Multimodal_pretraining/data/{data}/{data}_train_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'
    save_merged_adata(adata_merged=adata_merged, filename=filename)

    filename = f'./Multimodal_pretraining/data/{data}/{data}_test_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'
    save_merged_adata(adata_merged=adata_merged_test, filename=filename)

def test_r():
    adata_merged_train = sc.read_h5ad("./Multimodal_pretraining/data/simulated/simulated_train_1_mt_0_bs_64_77_0.001_0.1_False_True_128.h5ad")
    adata_merged = sc.read_h5ad("./Multimodal_pretraining/data/simulated/simulated_test_1_mt_0_bs_64_77_0.001_0.1_False_True_128.h5ad")
    res = query_to_reference(X_train=adata_merged_train.obsm[f'train_64_encoder_0.1'], 
                             y_train=adata_merged_train.obs["cell_type_l1"], 
                             X_test=adata_merged.obsm[f'test_64_encoder_0.1'], 
                             y_test=adata_merged.obs["cell_type_l1"])
    
    print(res)

    adata_merged.obsm[f'pred_cell_type_1'] = res.set_index(adata_merged.obs_names)["val_ct"]
    print(adata_merged)

main()
# test_r()
