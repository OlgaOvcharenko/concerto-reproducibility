import os
import sys

import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
sys.path.append("../")
from concerto_function5_3 import *
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
# Inital setting for plot size
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, silhouette_samples
import tensorflow as tf
from sklearn.metrics import confusion_matrix

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
    parser.add_argument('--task', type= int, required=True,
                        help='0-bc, 1-qr+mp')

    args = parser.parse_args()
    return args

def prepare_data_PBMC_together(train: bool = True, save_path: str = ''):
    print("Read PBMC data.")
    adata_RNA = sc.read_h5ad(save_path + f'adata_RNA_train.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_Protein_train.h5ad')

    adata_RNA_test = sc.read_h5ad(save_path + f'adata_RNA_test.h5ad')
    adata_Protein_test = sc.read_h5ad(save_path + f'adata_Protein_test.h5ad')

    print(f"RNA data shape train {adata_RNA.shape}, test {adata_RNA_test.shape}")
    print(f"Protein data shape {adata_Protein.shape}, test {adata_Protein_test.shape}")

    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]

    adata_merged_test = ad.concat([adata_RNA_test, adata_Protein_test], axis=1)
    sc.tl.pca(adata_merged_test)
    adata_merged_test.obsm["Unintegrated_HVG_only"] = adata_merged_test.obsm["X_pca"]
    
    print("Preprocessed data.")

    path_file = 'tfrecord_train/'
    RNA_tf_path = save_path + path_file + 'RNA_tf/'
    Protein_tf_path = save_path + path_file + 'Protein_tf/'

    path_file = 'tfrecord_test/'
    RNA_tf_path_test = save_path + path_file + 'RNA_tf/'
    Protein_tf_path_test = save_path + path_file + 'Protein_tf/'

    return RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_merged_test, adata_RNA_test

def prepare_data_PBMC_full(train: bool = True, save_path: str = ''):
    print("Read PBMC data.")
    adata_RNA = sc.read_h5ad(save_path + f'adata_RNA_full.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_Protein_full.h5ad')

    print(f"RNA data shape train {adata_RNA.shape}")
    print(f"Protein data shape {adata_Protein.shape}")

    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]
    
    print("Preprocessed data.")

    path_file = 'tfrecord_full/'
    RNA_tf_path = save_path + path_file + 'RNA_tf/'
    Protein_tf_path = save_path + path_file + 'Protein_tf/'

    return RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA

def prepare_data_neurips_cite_full(train: bool = True, save_path: str = ''):
    print("Read human data")
    adata_RNA = sc.read_h5ad(save_path + f'adata_neurips_GEX_full.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_neurips_ADT_full.h5ad')

    print(f"GEX data shape train {adata_RNA.shape}")
    print(f"ADT data shape train {adata_Protein.shape}")

    # Add PCA after preprocessing for benchmarking
    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]

    path_file = 'tfrecord_full/'
    RNA_tf_path = save_path + path_file + 'GEX_tf/'
    Protein_tf_path = save_path + path_file + 'ADT_tf/'

    return RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA

def prepare_data_neurips_multiome_full(train: bool = True, save_path: str = ''):
    print("Read human data")
    adata_RNA = sc.read_h5ad(save_path + f'adata_neurips_GEX_multiome_full.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_neurips_ATAC_multiome_full.h5ad')

    print(f"GEX data shape train {adata_RNA.shape}")
    print(f"ADT data shape train {adata_Protein.shape}")

    # Add PCA after preprocessing for benchmarking
    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]

    path_file = 'tfrecord_full/'
    RNA_tf_path = save_path + path_file + 'GEX_multiome_tf/'
    Protein_tf_path = save_path + path_file + 'ATAC_multiome_tf/'

    return RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA

def prepare_data_neurips_cite_together(train: bool = True, save_path: str = ''):
    print("Read human data")
    adata_RNA = sc.read_h5ad(save_path + f'adata_GEX_train.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_ADT_train.h5ad')

    adata_RNA_test = sc.read_h5ad(save_path + f'adata_GEX_test.h5ad')
    adata_Protein_test = sc.read_h5ad(save_path + f'adata_ADT_test.h5ad')

    print(f"GEX data shape train {adata_RNA.shape}, test {adata_RNA_test.shape}")
    print(f"ADT data shape train {adata_Protein.shape}, test {adata_Protein_test.shape}")

    # Add PCA after preprocessing for benchmarking
    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]

    adata_merged_test = ad.concat([adata_RNA_test, adata_Protein_test], axis=1)
    sc.tl.pca(adata_merged_test)
    adata_merged_test.obsm["Unintegrated_HVG_only"] = adata_merged_test.obsm["X_pca"]

    print("Saved adata.")

    path_file = 'tfrecord_train/'
    RNA_tf_path = save_path + path_file + 'GEX_tf/'
    Protein_tf_path = save_path + path_file + 'ADT_tf/'

    path_file = 'tfrecord_test/'
    RNA_tf_path_test = save_path + path_file + 'GEX_tf/'
    Protein_tf_path_test = save_path + path_file + 'ADT_tf/'
    return RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_merged_test, adata_RNA_test

def prepare_data_neurips_multiome_together(train: bool = True, save_path: str = ''):
    print("Read human data")
    adata_RNA = sc.read_h5ad(save_path + f'adata_GEX_multiome_train.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_ATAC_multiome_train.h5ad')

    adata_RNA_test = sc.read_h5ad(save_path + f'adata_GEX_multiome_test.h5ad')
    adata_Protein_test = sc.read_h5ad(save_path + f'adata_ATAC_multiome_test.h5ad')

    print(f"GEX data shape train {adata_RNA.shape}, test {adata_RNA_test.shape}")
    print(f"ADT data shape train {adata_Protein.shape}, test {adata_Protein_test.shape}")

    # Add PCA after preprocessing for benchmarking
    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]

    adata_merged_test = ad.concat([adata_RNA_test, adata_Protein_test], axis=1)
    sc.tl.pca(adata_merged_test)
    adata_merged_test.obsm["Unintegrated_HVG_only"] = adata_merged_test.obsm["X_pca"]

    print("Saved adata.")

    path_file = 'tfrecord_train/'
    RNA_tf_path = save_path + path_file + 'GEX_multiome_tf/'
    Protein_tf_path = save_path + path_file + 'ATAC_multiome_tf/'

    path_file = 'tfrecord_test/'
    RNA_tf_path_test = save_path + path_file + 'GEX_multiome_tf/'
    Protein_tf_path_test = save_path + path_file + 'ATAC_multiome_tf/'
    return RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_merged_test, adata_RNA_test


def read_data(data: str = "simulated", save_path: str = "", task=0):
    if data == "simulated":
        if task == 0:
            RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA = prepare_data_PBMC_full(train=True, save_path=save_path)
        else:
            RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_merged_test, adata_RNA_test = prepare_data_PBMC_together(train=True, save_path=save_path)
    
    elif data == "human_cite":
        if task == 0:
            RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA = prepare_data_neurips_cite_full(train=True, save_path=save_path)
        else:
            RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_merged_test, adata_RNA_test = prepare_data_neurips_cite_together(train=True, save_path=save_path)
    
    elif data == "human_multiome":
        if task == 0:
            RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA = prepare_data_neurips_multiome_full(train=True, save_path=save_path)
        else:
            RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_merged_test, adata_RNA_test = prepare_data_neurips_multiome_together(train=True, save_path=save_path)
    
    if task == 0:
        return RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA
    else:
        return RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_merged_test, adata_RNA_test

def train_concerto(weight_path: str, RNA_tf_path: str, Protein_tf_path: str, data: str, 
                   attention_t: bool, attention_s: bool,
                   batch_size:int, epoch: int, lr: float, drop_rate: float, 
                   heads: int, combine_omics: int, model_type: int, task: int = 0):
    if attention_t == True and attention_s == False:
        concerto_train_multimodal(['RNA','Protein'] if data == 'simulated' else ['ATAC', 'GEX'] if data == 'human' else ["GEX", "ADT"],
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
                                    'model_type': model_type,
                                    'task': task
                                    })
    else:
        raise Exception("Invalid Teacher/Student combination.")

    print("Trained.")

def test_concerto_qr(adata_merged, adata_RNA, weight_path: str, RNA_tf_path_test: str, Protein_tf_path_test: str, data: str, 
                   attention_t: bool, attention_s: bool,
                   batch_size:int, epoch: int, lr: float, drop_rate: float, 
                   heads: int, combine_omics: int, model_type: int, 
                   save_path: str, train: bool = False, adata_merged_train = None, repeat: int = 0, task: int = 0):
    # adata_merged.obs = adata_RNA.obs

    # Test
    nn = "encoder"
    dr = 0.0 # drop_rate
    only_RNA = False
    e = epoch
    accuracies, macro_f1s, weighted_f1s, per_class_f1s, median_f1s = [], [], [], [], []
    
    saved_weight_path = f'./Multimodal_pretraining/weight/multi_weight_{nn}_{data}_{batch_size}_model_{combine_omics}_{model_type}_epoch_{e}_{lr}_{drop_rate}_{attention_t}_{attention_s}_{heads}.h5'
    embedding, _, RNA_id, _ =  concerto_test_multimodal(
            ['RNA','Protein'] if data == 'simulated' else ['ATAC', 'GEX'] if data == 'human' else ["GEX", "ADT"],
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
                'model_type': model_type,
                'data' : data,
                'task': task,
            }, 
            saved_weight_path = saved_weight_path,
            only_RNA=only_RNA)
    
    if data == "simulated":
        adata_RNA = sc.read(save_path + f'adata_RNA_{"train" if train else "test"}.h5ad')
    elif data == 'human_cite':
        adata_RNA = sc.read(save_path + f'adata_GEX_{"train" if train else "test"}.h5ad')
    elif data == 'human_multiome':
        adata_RNA = sc.read(save_path + f'adata_GEX_multiome_{"train" if train else "test"}.h5ad')
    
    adata_RNA_1 = adata_RNA[RNA_id]
    adata_RNA_1.obsm['X_embedding'] = embedding
    adata_merged = adata_merged[RNA_id]
    adata_merged.obsm[f'train_{e}_{nn}_{dr}_{only_RNA}_{repeat}' if train else f'test_{e}_{nn}_{dr}_{only_RNA}_{repeat}'] = embedding
    # print(f"\nShape of the {train}_{e}_{nn}_{dr}_{only_RNA}_{repeat} embedding {embedding.shape}.")
    
    if not train:
        # Reference query mapping
        query_neighbor, _ = knn_classifier(ref_embedding=adata_merged_train.obsm[f'train_{e}_{nn}_{dr}_{only_RNA}_{repeat}'], query_embedding=embedding, ref_anndata=adata_merged_train, column_name='cell_type_l1', k=5)

        cell_types_list = pd.unique(adata_merged.obs['cell_type_l1']).tolist()
        acc = accuracy_score(adata_merged.obs['cell_type_l1'].to_list(), query_neighbor)
        f1 = f1_score(adata_merged.obs['cell_type_l1'].to_list(), query_neighbor, labels=cell_types_list, average=None)
        f1_weighted = f1_score(adata_merged.obs['cell_type_l1'].to_list(), query_neighbor, labels=cell_types_list, average='weighted')
        f1_macro = f1_score(adata_merged.obs['cell_type_l1'].to_list(), query_neighbor, labels=cell_types_list, average='macro')
        f1_median = np.median(f1)
        
        print(f"Per class {cell_types_list} F1 {f1}")
        print('Accuracy {:.3f}, F1 median {:.3f}, F1 macro {:.3f}, F1 weighted {:.3f} '.format(acc, f1_median, f1_macro, f1_weighted),)

        return adata_merged, acc, f1_median, f1_macro, f1_weighted

        # Missing modality prediction
    return adata_merged

def test_concerto_mp(weight_path: str, RNA_tf_path: str, Protein_tf_path: str,
                     RNA_tf_path_test: str, Protein_tf_path_test: str, data: str, 
                     attention_t: bool, attention_s: bool,
                     batch_size:int, epoch: int, lr: float, drop_rate: float, 
                     heads: int, combine_omics: int, model_type: int, 
                     save_path: str, repeat: int = 0, task: int = 0):
    # Test
    nn = "encoder"
    dr = 0.0 # drop_rate
    only_RNA = False
    e = epoch
    
    saved_weight_path = f'./Multimodal_pretraining/weight/multi_weight_{nn}_{data}_{batch_size}_model_{combine_omics}_{model_type}_epoch_{e}_{lr}_{drop_rate}_{attention_t}_{attention_s}_{heads}.h5'
    
    embedding_train, embedding2_train, _, RNA_id_train, _ =  concerto_test_multimodal_modalities(
            ['RNA','Protein'] if data == 'simulated' else ['ATAC', 'GEX'] if data == 'human' else ["GEX", "ADT"],
            weight_path, 
            RNA_tf_path,
            Protein_tf_path,
            n_cells_for_sample=None,
            super_parameters={
                'batch_size': batch_size, 
                'epoch_pretrain': e, 'lr': lr, 
                'drop_rate': dr, 
                'attention_t': attention_t, 
                'attention_s': attention_s, 
                'heads': heads,
                'combine_omics': combine_omics,
                'model_type': model_type,
                'data' : data
            }, 
            saved_weight_path = saved_weight_path,
            only_RNA=only_RNA)
    
    embedding, embedding2, _, RNA_id, _ =  concerto_test_multimodal_modalities(
            ['RNA','Protein'] if data == 'simulated' else ['ATAC', 'GEX'] if data == 'human' else ["GEX", "ADT"],
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
                'model_type': model_type,
                'data' : data,
                'task':task,
            }, 
            saved_weight_path = saved_weight_path,
            only_RNA=only_RNA)
    
    if data == "simulated":
        adata_Protein = sc.read(save_path + f'adata_Protein_test.h5ad')
        adata_Protein_train = sc.read(save_path + f'adata_Protein_train.h5ad')
    elif data == 'human_cite':
        adata_Protein = sc.read(save_path + f'adata_ADT_test.h5ad')
        adata_Protein_train = sc.read(save_path + f'adata_ADT_train.h5ad')
    elif data == 'human_multiome':
        adata_Protein = sc.read(save_path + f'adata_ATAC_multiome_test.h5ad')
        adata_Protein_train = sc.read(save_path + f'adata_ATAC_multiome_train.h5ad')
    
    nbrs = NearestNeighbors(metric='cosine', n_neighbors=5, algorithm='auto').fit(embedding_train)
    indices = nbrs.kneighbors(embedding, return_distance=False)

    print(indices.shape)
    print(adata_Protein_train.shape)

    # slice acc to embedding
    adata_Protein = adata_Protein[RNA_id]
    adata_Protein_train = adata_Protein_train[RNA_id_train]

    val_new_protein = np.array(adata_Protein_train.X.todense())[indices].mean(axis=1)
    tmp = adata_Protein.X.todense()

    pearsons = []
    for true_protein, pred_protein in zip(tmp, val_new_protein):
        pearsons.append(np.corrcoef(pred_protein, true_protein)[0, 1])

    print(f'Pearson {repeat}: {np.mean(pearsons)}')

    return np.mean(pearsons)

def test_concerto_bc(adata_merged, adata_RNA, weight_path: str, RNA_tf_path_test: str, Protein_tf_path_test: str, data: str, 
                     attention_t: bool, attention_s: bool, batch_size:int, epoch: int, lr: float, drop_rate: float, 
                     heads: int, combine_omics: int, model_type: int, save_path: str, train: bool = False, repeat: int = 0, task: int = 0):
    adata_merged.obs = adata_RNA.obs

    # Test
    nn = "encoder"
    dr = 0.0 # drop_rate
    e = epoch
    only_RNA = False
    
    saved_weight_path = f'./Multimodal_pretraining/weight/multi_weight_{nn}_{data}_{batch_size}_model_{combine_omics}_{model_type}_epoch_{e}_{lr}_{drop_rate}_{attention_t}_{attention_s}_{heads}_{task}.h5'
    
    embedding, batch, RNA_id, attention_weight =  concerto_test_multimodal(
            ['RNA','Protein'] if data == 'simulated' else ['ATAC', 'GEX'] if data == 'human' else ["GEX", "ADT"],
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
                'model_type': model_type,
                'data' : data,
                'task':task,
            }, 
            saved_weight_path = saved_weight_path,
            only_RNA=only_RNA)
    
    if data == "simulated":
        adata_RNA = sc.read(save_path + f'adata_RNA_full.h5ad')
    elif data == 'human_cite':
        adata_RNA = sc.read(save_path + f'adata_neurips_GEX_full.h5ad')
    elif data == 'human_multiome':
        adata_RNA = sc.read(save_path + f'adata_neurips_GEX_multiome_full.h5ad')
    adata_RNA_1 = adata_RNA[RNA_id]
    adata_RNA_1.obsm['X_embedding'] = embedding

    print(f"\nShape of the {train}_{e}_{nn}_{dr}_{only_RNA}_{repeat} embedding {embedding.shape}.")
        
    adata_merged = adata_merged[RNA_id]
    adata_merged.obsm[f'{e}_{nn}_{dr}_{only_RNA}_{repeat}'] = embedding
            
    return adata_merged

def save_merged_adata(adata_merged, filename):
    adata_merged.write(filename)

    print(adata_merged)
    print(f"Saved adata all at {filename}")

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
    task = args.task

    print(f"Multimodal correction: epoch {epoch}, model type {model_type}, lr {lr}, batch_size {batch_size}, drop_rate {drop_rate}, attention_t {attention_t}, attention_s {attention_s}, heads {heads}, task {task}.")

    # Check num GPUs
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(f"\nAvailable GPUs: {gpus}\n")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    
    # Read data
    save_path = './Multimodal_pretraining/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    res_df = pd.DataFrame(columns=["epoch", "accuracy", "f1_median", "f1_macro", "f1_weighted", "pearson" ])
    for repeat in range(0, 1):
        if task == 0:
            RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA = read_data(data=data, save_path=save_path, task=task)
        elif task == 1:
            RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_merged_test, adata_RNA_test = read_data(data=data, save_path=save_path, task=task)

        # Train
        weight_path = save_path + 'weight/'
        if train:
            train_concerto(weight_path=weight_path, RNA_tf_path=RNA_tf_path, Protein_tf_path=Protein_tf_path, data=data, 
                    attention_t=attention_t, attention_s=attention_s, 
                    batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
                    heads=heads, combine_omics=combine_omics, model_type=model_type, task=task)
        print("Trained.")

        if test:
            if task == 0:
                print(f"repeat {repeat}")
                adata_merged = test_concerto_bc(weight_path=weight_path, RNA_tf_path_test=RNA_tf_path, Protein_tf_path_test=Protein_tf_path, data=data, 
                        attention_t=attention_t, attention_s=attention_s,
                        batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
                        heads=heads, combine_omics=combine_omics, model_type=model_type, 
                        save_path=save_path, train=True, adata_merged=adata_merged, adata_RNA=adata_RNA, repeat=repeat, task=task)
                
                filename = f'./Multimodal_pretraining/data/{data}/{data}_bc_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}_{repeat}.h5ad'
                save_merged_adata(adata_merged=adata_merged, filename=filename)

            else:
                # Query-to-reference
                # Test on train data
                i, ep_vals = 8, []
                while i < epoch:
                    ep_vals.append(i)
                    i = i * 2
                ep_vals.append(epoch)

                adata_merged.obs = adata_RNA.obs
                adata_merged_test.obs = adata_RNA_test.obs

                for e in ep_vals:
                    adata_merged = test_concerto_qr(weight_path=weight_path, RNA_tf_path_test=RNA_tf_path, Protein_tf_path_test=Protein_tf_path, data=data, 
                            attention_t=attention_t, attention_s=attention_s,
                            batch_size=batch_size, epoch=e, lr=lr, drop_rate=drop_rate, 
                            heads=heads, combine_omics=combine_omics, model_type=model_type, 
                            save_path=save_path, train=True, adata_merged=adata_merged, adata_RNA=adata_RNA, repeat=repeat, task=task)
                    
                    filename = f'./Multimodal_pretraining/data/{data}/{data}_qr_train_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}_{repeat}.h5ad'
                    save_merged_adata(adata_merged=adata_merged, filename=filename)

                    # Test on test data
                    adata_merged_test, acc, f1_median, f1_macro, f1_weighted = test_concerto_qr(weight_path=weight_path, RNA_tf_path_test=RNA_tf_path_test, Protein_tf_path_test=Protein_tf_path_test, data=data, 
                            attention_t=attention_t, attention_s=attention_s,
                            batch_size=batch_size, epoch=e, lr=lr, drop_rate=drop_rate, 
                            heads=heads, combine_omics=combine_omics, model_type=model_type, 
                            save_path=save_path, train=False, adata_merged=adata_merged_test, adata_RNA=adata_RNA_test, adata_merged_train=adata_merged, repeat=repeat, task=task)

                    filename = f'./Multimodal_pretraining/data/{data}/{data}_qr_test_{combine_omics}_mt_{model_type}_bs_{batch_size}_{e}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}_{repeat}.h5ad'
                    save_merged_adata(adata_merged=adata_merged_test, filename=filename)

                    # # Model prediction
                    # pearson = test_concerto_mp(weight_path=weight_path, data=data, 
                    #                  RNA_tf_path_test=RNA_tf_path_test, Protein_tf_path_test=Protein_tf_path_test, 
                    #                  RNA_tf_path=RNA_tf_path, Protein_tf_path=Protein_tf_path, 
                    #                  attention_t=attention_t, attention_s=attention_s,
                    #                  batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
                    #                  heads=heads, combine_omics=combine_omics, model_type=model_type, 
                    #                  save_path=save_path, repeat=repeat)
                    
                    res_df.loc[repeat] = [e, acc, f1_median, f1_macro, f1_weighted, 0.0]
    
    if task != 0:
        res_df.to_csv(f'./Multimodal_pretraining/results/{data}/{data}_qr_train_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.csv')

main()
# test_r()
