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
    adata_RNA = adata_RNA.read_h5ad(save_path + f'adata_RNA_train.h5ad')
    adata_Protein = adata_Protein.write_h5ad(save_path + f'adata_Protein_train.h5ad')

    adata_RNA_test = adata_RNA.read_h5ad(save_path + f'adata_RNA_test.h5ad')
    adata_Protein_test = adata_Protein.write_h5ad(save_path + f'adata_Protein_test.h5ad')

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
    adata_RNA = adata_RNA.read_h5ad(save_path + f'adata_RNA_full.h5ad')
    adata_Protein = adata_Protein.write_h5ad(save_path + f'adata_Protein_full.h5ad')

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
    adata_RNA = adata_RNA.read_h5ad(save_path + f'adata_GEX_full.h5ad')
    adata_Protein = adata_Protein.read_h5ad(save_path + f'adata_ADT_full.h5ad')

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

def prepare_data_neurips_cite_together(train: bool = True, save_path: str = ''):
    print("Read human data")
    adata_RNA = adata_RNA.read_h5ad(save_path + f'adata_GEX_train.h5ad')
    adata_Protein = adata_Protein.read_h5ad(save_path + f'adata_ADT_train.h5ad')

    adata_RNA_test = adata_RNA.read_h5ad(save_path + f'adata_GEX_test.h5ad')
    adata_Protein_test = adata_Protein.read_h5ad(save_path + f'adata_ADT_test.h5ad')

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
    
    if task == 0:
        return RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA
    else:
        return RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_merged_test, adata_RNA_test

def train_concerto(weight_path: str, RNA_tf_path: str, Protein_tf_path: str, data: str, 
                   attention_t: bool, attention_s: bool,
                   batch_size:int, epoch: int, lr: float, drop_rate: float, 
                   heads: int, combine_omics: int, model_type: int):
    # Train
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
                                    'model_type': model_type
                                    })
    elif attention_t == True and attention_s == True:
        concerto_train_multimodal_tt(['RNA','Protein'] if data == 'simulated' else ['ATAC', 'GEX'] if data == 'human' else ["GEX", "ADT"],
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
        concerto_train_multimodal_ss(['RNA','Protein'] if data == 'simulated' else ['ATAC', 'GEX'] if data == 'human' else ["GEX", "ADT"],
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
    nn = "encoder"
    dr = 0.0 # drop_rate
    only_RNAs = [True, False] if combine_omics == 0 else [False]
    repeats = [0] if not train else [0]
    for only_RNA in only_RNAs:
        for e in ep_vals: 
            saved_weight_path = f'./Multimodal_pretraining/weight/multi_weight_{nn}_{data}_{batch_size}_model_{combine_omics}_{model_type}_epoch_{e}_{lr}_{drop_rate}_{attention_t}_{attention_s}_{heads}.h5'
            accuracies, macro_f1s, weighted_f1s, per_class_f1s, median_f1s = [], [], [], [], []

            for repeat in repeats:
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
                            'data' : data
                        }, 
                        saved_weight_path = saved_weight_path,
                        only_RNA=only_RNA)
                
                if data == "simulated":
                    adata_RNA = sc.read(save_path + f'adata_RNA_{"train" if train else "test"}.h5ad')
                elif data == 'human':
                    adata_RNA = sc.read(save_path + f'adata_atac_{"train" if train else "test"}.h5ad')
                elif data == 'human_cite':
                    adata_RNA = sc.read(save_path + f'adata_cite_gex_{"train" if train else "test"}.h5ad')
                elif data == 'human_cite_raw':
                    adata_RNA = sc.read(save_path + f'adata_cite_gex_unprep_{"train" if train else "test"}.h5ad')
                
                adata_RNA_1 = adata_RNA[RNA_id]
                adata_RNA_1.obsm['X_embedding'] = embedding

                print(f"\nShape of the {train}_{e}_{nn}_{dr}_{only_RNA}_{repeat} embedding {embedding.shape}.")
                
                adata_merged = adata_merged[RNA_id]
                adata_merged.obsm[f'train_{e}_{nn}_{dr}_{only_RNA}' if train else f'test_{e}_{nn}_{dr}_{only_RNA}_{repeat}'] = embedding
                
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
                    query_neighbor, _ = knn_classifier(ref_embedding=adata_merged_train.obsm[f'train_{e}_{nn}_{dr}_{only_RNA}'], query_embedding=embedding, ref_anndata=adata_merged_train, column_name='cell_type_l1', k=5)
                    adata_RNA_1.obs[f'pred_cell_type_{e}_{nn}_{dr}_{only_RNA}_{repeat}'] = query_neighbor
                    adata_merged.obs[f'pred_cell_type_{e}_{nn}_{dr}_{only_RNA}_{repeat}'] = query_neighbor
                    
                    cell_types_list = pd.unique(adata_merged.obs['cell_type_l1']).tolist()
                    acc = accuracy_score(adata_merged.obs['cell_type_l1'].to_list(), query_neighbor)
                    f1 = f1_score(adata_merged.obs['cell_type_l1'].to_list(), query_neighbor, labels=cell_types_list, average=None)
                    f1_weighted = f1_score(adata_merged.obs['cell_type_l1'].to_list(), query_neighbor, labels=cell_types_list, average='weighted')
                    f1_macro = f1_score(adata_merged.obs['cell_type_l1'].to_list(), query_neighbor, labels=cell_types_list, average='macro')
                    f1_median = np.median(f1)

                    accuracies.append(acc)
                    macro_f1s.append(f1_macro)
                    weighted_f1s.append(f1_weighted)
                    per_class_f1s.append(f1)
                    median_f1s.append(f1_median)
                    
                    print(f"Per class {cell_types_list} F1 {f1}")
                    print('Accuracy {:.3f}, F1 median {:.3f}, F1 macro {:.3f}, F1 weighted {:.3f} '.format(acc, f1_median, f1_macro, f1_weighted),)


                sc.tl.leiden(adata_RNA_1, resolution=0.2)
                sc.tl.umap(adata_RNA_1, min_dist=0.1)

                adata_merged.obsm[f'train_umap_{e}_{nn}_{dr}_{only_RNA}' if train else f'test_umap_{e}_{nn}_{dr}_{only_RNA}_{repeat}'] = adata_RNA_1.obsm["X_umap"]
                adata_merged.obs[f'train_leiden_{e}_{nn}_{dr}_{only_RNA}' if train else f'test_leiden_{e}_{nn}_{dr}_{only_RNA}_{repeat}'] = adata_RNA_1.obs["leiden"]

                if not train:
                    color=['cell_type_l1', f'pred_cell_type_{e}_{nn}_{dr}_{only_RNA}_{repeat}', 'leiden', 'batch']
                else:
                    color=['cell_type_l1', 'leiden', 'batch']

                sc.set_figure_params(dpi=150)
                sc.pl.umap(adata_RNA_1, color=color, legend_fontsize ='xx-small', size=5, legend_fontweight='light') # edges=True
                plt.savefig(f'./Multimodal_pretraining/plots/{data}/{data}_knn_concerto_{"train" if train else "test"}_{combine_omics}_oGEX{only_RNA}_mt_{model_type}_bs_{batch_size}_{nn}_{e}_{lr}_{drop_rate}_{dr}_{attention_s}_{attention_t}_{heads}_repeat{repeat}.png')

            if not train:
                print(f"\nShape of the {e}_{nn}_{dr}_{only_RNA}_{repeat} embedding {embedding.shape}.")
                print(f"Averaged per class {adata_merged.obs['cell_type_l1'].to_list()} F1 {np.array(per_class_f1s).mean(axis=0)}")
                print('Averaged after 5 runs accuracy {:.3f}, F1 median {:.3f}, F1 macro {:.3f}, F1 weighted {:.3f} '.format(np.mean(accuracies), np.mean(median_f1s), np.mean(macro_f1s), np.mean(weighted_f1s)),)
            
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
    RNA_tf_path, Protein_tf_path, adata_merged, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_merged_test, adata_RNA_test = read_data(data=data, save_path=save_path, task=task)

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
        
        filename = f'./Multimodal_pretraining/data/{data}/{data}_train_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'
        save_merged_adata(adata_merged=adata_merged, filename=filename)

        # Test on test data
        adata_merged_test = test_concerto(weight_path=weight_path, RNA_tf_path_test=RNA_tf_path_test, Protein_tf_path_test=Protein_tf_path_test, data=data, 
                   attention_t=attention_t, attention_s=attention_s,
                   batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
                   heads=heads, combine_omics=combine_omics, model_type=model_type, 
                   save_path=save_path, train=False, adata_merged=adata_merged_test, adata_RNA=adata_RNA_test, adata_merged_train=adata_merged)

        filename = f'./Multimodal_pretraining/data/{data}/{data}_test_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'
        save_merged_adata(adata_merged=adata_merged_test, filename=filename)

main()
# test_r()
