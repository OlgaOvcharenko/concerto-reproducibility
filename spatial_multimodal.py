import argparse
import math
import os
import sys

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
sys.path.append("../")
from concerto_function5_3 import *
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib

l2tol1 ={
    "alveolar cells type 1": "epithelial", 
    "alveolar cells type 2": "epithelial", 
    "ciliated": "epithelial",
    "endothelial": "endothelial", 
    "lymphatic endothelial": "endothelial", 
    "capillary":"endothelial",
    "mesothelial": "mesothelial",
    "fibroblast": "structural", 
    "smooth muscle": "structural",
    "tumor": "tumor",
    "neutrophil": "granulocyte", 
    "granulocyte": "granulocyte",
    "macrophage": "phagocyte", 
    "monocyte": "phagocyte", 
    "dendritic": "phagocyte",
    "b cell": "lymphocyte", 
    "t cell": "lymphocyte", 
    "plasma": "lymphocyte",
    "NA": "other"
}

SUPERCATEGORY_ORDER = ["epithelial", "endothelial", "mesothelial", "structural", "tumor", "granulocyte", "phagocyte", "lymphocyte", "other"]

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
    
    parser.add_argument('--mask', type= int, required=True,
                        help='0/1')
    parser.add_argument('--model_type_image', type= int, required=True,
                        help='0 VGG+CNN, 1 EfficientNetB4+B7')

    args = parser.parse_args()
    return args

def train_concerto(weight_path: str, RNA_tf_path: str, staining_tf_path: str, data: str, 
                   attention_t: bool, attention_s: bool,
                   batch_size:int, epoch: int, lr: float, drop_rate: float, 
                   heads: int, combine_omics: int, model_type: int, mask: bool, model_type_image: int = 0):
    # Train
    concerto_train_spatial_multimodal(['RNA','staining'], 
                                      RNA_tf_path, 
                                      staining_tf_path, 
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
                                          'mask': mask,
                                          'model_type_image': model_type_image,
                                          })

    print("Trained.")

def test_concerto(adata_RNA, weight_path: str, data: str, 
                  RNA_tf_path_test: str, staining_tf_path: str, 
                  attention_t: bool, attention_s: bool,
                  batch_size:int, epoch: int, lr: float, drop_rate: float, 
                  heads: int, combine_omics: int, model_type: int, mask: int,
                  save_path: str, train: bool = False, adata_RNA_train = None, model_type_image: int = 0):
    ep_vals = []
    i = 32 # i = 4
    while i < epoch:
        ep_vals.append(i)
        i = i * 2
    ep_vals.append(epoch)

    adata_merged = adata_RNA.copy()

    # Test
    nn = "encoder"
    dr = 0.0 # drop_rate
    only_images = [False] # if combine_omics == 0 else [False]
    for only_image in only_images:
        for e in [200]: 
            saved_weight_path = f'./Multimodal_pretraining/weight/multi_weight_{nn}_{data}_{mask}_{batch_size}_model_{combine_omics}_{model_type}_epoch_{e}_{lr}_{drop_rate}_{attention_t}_{attention_s}_{heads}.h5'
            
            embedding, _, RNA_id =  concerto_test_spatial_multimodal(
                    ['RNA', 'staining'],
                    weight_path, 
                    RNA_tf_path_test,
                    staining_tf_path,
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
                        'mask': mask,
                        'model_type_image': model_type_image,
                    }, 
                    saved_weight_path = saved_weight_path,
                    only_image=only_image)
            
            adata_RNA_1 = adata_RNA[RNA_id].copy()
            adata_RNA_1.obsm['X_embedding'] = embedding

            print(f"\nShape of the {train}_{e}_{nn}_{dr}_{only_image} embedding {embedding.shape}.")
            
            print(adata_merged)
            adata_merged = adata_RNA[RNA_id]
            adata_merged.obsm[f'train_{e}_{nn}_{dr}_{only_image}' if train else f'test_{e}_{nn}_{dr}_{only_image}'] = embedding
            print(adata_merged)

            sc.pp.neighbors(adata_RNA_1, use_rep="X_embedding", metric="cosine")
            labels = adata_RNA_1.obs['cell_type'].tolist()
            for res in [0.05,0.1,0.15,0.2,0.25,0.3]:
                sc.tl.leiden(adata_RNA_1, resolution=res)
                target_preds = adata_RNA_1.obs['leiden'].tolist()
                nmi = np.round(normalized_mutual_info_score(labels, target_preds), 5)
                ari = np.round(adjusted_rand_score(labels, target_preds), 5)    
                n_cluster = len(list(set(target_preds)))
                print('leiden(res=%f): ari = %.5f , nmi = %.5f, n_cluster = %d' % (res, ari, nmi, n_cluster), '.')

            if not train:
                query_neighbor, _ = knn_classifier(ref_embedding=adata_RNA_train.obsm[f'train_{e}_{nn}_{dr}_{only_image}'], query_embedding=embedding, ref_anndata=adata_RNA_train, column_name='cell_type', k=5)
                adata_RNA_1.obs[f'pred_cell_type_{e}_{nn}_{dr}_{only_image}'] = query_neighbor
                adata_merged.obs[f'pred_cell_type_{e}_{nn}_{dr}_{only_image}'] = query_neighbor
                
                cell_types_list = pd.unique(adata_merged.obs['cell_type']).tolist()

                acc = accuracy_score(adata_merged.obs['cell_type'].to_list(), query_neighbor)
                f1 = f1_score(adata_merged.obs['cell_type'].to_list(), query_neighbor, labels=cell_types_list, average=None)
                f1_weighted = f1_score(adata_merged.obs['cell_type'].to_list(), query_neighbor, labels=cell_types_list, average='weighted')
                f1_macro = f1_score(adata_merged.obs['cell_type'].to_list(), query_neighbor, labels=cell_types_list, average='macro')
                f1_median = np.median(f1)
                
                print(f"Per class {cell_types_list} F1 {f1}")
                print('Accuracy {:.3f}, F1 median {:.3f}, F1 macro {:.3f}, F1 weighted {:.3f} '.format(acc, f1_median, f1_macro, f1_weighted),)


            sc.tl.leiden(adata_RNA_1, resolution=0.2)
            sc.tl.umap(adata_RNA_1, min_dist=0.1)

            adata_merged.obsm[f'train_umap_{e}_{nn}_{dr}_{only_image}' if train else f'test_umap_{e}_{nn}_{dr}_{only_image}'] = adata_RNA_1.obsm["X_umap"]
            adata_merged.obs[f'train_leiden_{e}_{nn}_{dr}_{only_image}' if train else f'test_leiden_{e}_{nn}_{dr}_{only_image}'] = adata_RNA_1.obs["leiden"]

            if not train:
                color=['cell_type', f'pred_cell_type_{e}_{nn}_{dr}_{only_image}']
            else:
                color=['cell_type']
            # sc.set_figure_params(dpi=150)
            sc.pl.umap(adata_RNA_1, color=color, size=10, legend_fontweight='light') # edges=True
            plt.savefig(f'./Multimodal_pretraining/plots/{data}/{data}_{mask}_{"train" if train else "test"}_{combine_omics}_oRNA{only_image}_mt_{model_type}_bs_{batch_size}_{nn}_{e}_{lr}_{drop_rate}_{dr}_{attention_s}_{attention_t}_{heads}.png')

    return adata_merged

def test_concerto_full(adata_RNA, adata_RNA_test, weight_path: str, data: str, 
                  RNA_tf_path: str, staining_tf_path: str, 
                  RNA_tf_path_test: str, staining_tf_path_test: str, 
                  attention_t: bool, attention_s: bool,
                  batch_size:int, epoch: int, lr: float, drop_rate: float, 
                  heads: int, combine_omics: int, model_type: int, mask: int, 
                  cell_labels: str, model_type_image: int = 0):
    ep_vals = []
    i = 4
    while i < epoch:
        ep_vals.append(i)
        i = i * 2
    ep_vals.append(epoch)

    nn = "encoder"
    dr = 0.0 # drop_rate
    only_images = [False, True]

    r_i = 0
    res_df = pd.DataFrame(columns=["epoch", "onlyImage", "modality", "accuracy", "f1_median", "f1_macro", "f1_weighted"])
    for only_image in only_images:
        for e in ep_vals: 
            saved_weight_path = f'./Multimodal_pretraining/weight/multi_weight_{nn}_{data}_{mask}_{batch_size}_model_{combine_omics}_{model_type}_epoch_{e}_{lr}_{drop_rate}_{attention_t}_{attention_s}_{heads}.h5'
            
            ret_train = concerto_test_spatial_multimodal(
                    ['RNA', 'staining'],
                    weight_path, 
                    RNA_tf_path,
                    staining_tf_path,
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
                        'mask': mask,
                        'model_type_image': model_type_image
                    }, 
                    saved_weight_path = saved_weight_path,
                    only_image=only_image)

            ret_test = concerto_test_spatial_multimodal(
                    ['RNA', 'staining'],
                    weight_path, 
                    RNA_tf_path_test,
                    staining_tf_path_test,
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
                        'mask': mask,
                        'model_type_image': model_type_image
                    }, 
                    saved_weight_path = saved_weight_path,
                    only_image=only_image)
            
            if not only_image:
                embeddings = [ret_train[0]]
                RNA_id = ret_train[2]
                embedding_tests = [ret_test[0]]
                RNA_id_test = ret_test[2]
                modalities = ["both"]
            else: 
                embeddings = [ret_train[0], ret_train[1]]
                RNA_id = ret_train[3]
                embedding_tests = [ret_test[0], ret_test[1]]
                RNA_id_test = ret_test[3]
                modalities = ["RNA", "image"]

            for embedding, embedding_test, modality in zip(embeddings, embedding_tests, modalities):
                query_neighbor, _ = knn_classifier(ref_embedding=embedding, query_embedding=embedding_test, ref_anndata=adata_RNA[RNA_id], column_name=cell_labels, k=5)
                
                cell_types_list = SUPERCATEGORY_ORDER if cell_labels == 'cell_type_l1' else pd.unique(adata_RNA.obs[cell_labels]).tolist()

                acc = accuracy_score(adata_RNA_test[RNA_id_test].obs[cell_labels].to_list(), query_neighbor)
                f1 = f1_score(adata_RNA_test[RNA_id_test].obs[cell_labels].to_list(), query_neighbor, labels=cell_types_list, average=None)
                f1_weighted = f1_score(adata_RNA_test[RNA_id_test].obs[cell_labels].to_list(), query_neighbor, labels=cell_types_list, average='weighted')
                f1_macro = f1_score(adata_RNA_test[RNA_id_test].obs[cell_labels].to_list(), query_neighbor, labels=cell_types_list, average='macro')
                f1_median = np.median(f1)
                
                print(f"Modality {modality}, epoch {e}")
                print(f"Per class {cell_types_list} F1 {f1}")
                print('Accuracy {:.3f}, F1 median {:.3f}, F1 macro {:.3f}, F1 weighted {:.3f} '.format(acc, f1_median, f1_macro, f1_weighted))
                
                res_df.loc[r_i] = [e, only_image, modality, acc, f1_median, f1_macro, f1_weighted]
                r_i += 1

                # Add plot train and test
                adata_RNA_1 = adata_RNA[RNA_id].copy()
                adata_RNA_1.obsm['X_embedding'] = embedding
                sc.pp.neighbors(adata_RNA_1, use_rep="X_embedding", metric="cosine")
                sc.tl.umap(adata_RNA_1, min_dist=0.1)
                sc.pl.umap(adata_RNA_1, color=[cell_labels], size=10, legend_fontweight='light') # edges=True
                plt.savefig(f'./Multimodal_pretraining/plots/{data}/{data}_{mask}_train_{combine_omics}_oRNA{only_image}_mt_{model_type}_bs_{batch_size}_{nn}_{e}_{lr}_{drop_rate}_{dr}_{attention_s}_{attention_t}_{heads}.png')

                adata_RNA_1_test = adata_RNA_test[RNA_id_test].copy()
                adata_RNA_1_test.obsm['X_embedding'] = embedding_test
                adata_RNA_1_test.obs[f'pred_cell_type'] = query_neighbor
                sc.pp.neighbors(adata_RNA_1_test, use_rep="X_embedding", metric="cosine")
                sc.tl.umap(adata_RNA_1_test, min_dist=0.1)
                sc.pl.umap(adata_RNA_1_test, color=[cell_labels, 'pred_cell_type'], size=10, legend_fontweight='light') # edges=True
                plt.savefig(f'./Multimodal_pretraining/plots/{data}/{data}_{mask}_test_{combine_omics}_oRNA{only_image}_{modality}_mt_{model_type}_bs_{batch_size}_{nn}_{e}_{lr}_{drop_rate}_{dr}_{attention_s}_{attention_t}_{heads}.png')

    return res_df

def read_data_prepared(data: str = "", save_path: str = ""):
    print("Read data")
    print("Read spatial data.")
    adata_RNA = sc.read_h5ad(save_path + f'spatial_adata_RNA.h5ad')
    adata_RNA.obs['cell_type_l1'] = adata_RNA.obs['cell_type'].map(l2tol1)

    path_file = 'tfrecord/'
    RNA_tf_path = save_path + path_file + 'spatial_RNA_tf/'
    staining_tf_path = save_path + path_file + 'spatial_staining_tf/'
    
    return RNA_tf_path, adata_RNA, staining_tf_path

def read_data_split_prepared(data: str = "", save_path: str = ""):
    print("Read spatial data.")

    adata_RNA = sc.read_h5ad(save_path + f'train_spatial_adata_RNA.h5ad')
    adata_RNA_test = sc.read_h5ad(save_path + f'test_spatial_adata_RNA.h5ad')
    adata_RNA.obs['cell_type_l1'] = adata_RNA.obs['cell_type'].map(l2tol1)
    adata_RNA_test.obs['cell_type_l1'] = adata_RNA_test.obs['cell_type'].map(l2tol1)
    
    path_file = 'tfrecord/'
    RNA_tf_path = save_path + path_file + 'train_spatial_RNA_tf/'
    RNA_tf_path_test = save_path + path_file + 'test_spatial_RNA_tf/'

    staining_tf_path = save_path + path_file + 'train_spatial_staining_tf'
    staining_tf_path_test = save_path + path_file + 'test_spatial_staining_tf'
    
    return RNA_tf_path, adata_RNA, staining_tf_path, RNA_tf_path_test, adata_RNA_test, staining_tf_path_test


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
    mask = args.mask
    model_type_image = args.model_type_image
    cell_labels = 'cell_type_l1'

    print(f"Multimodal correction: epoch {epoch}, model type {model_type}, lr {lr}, batch_size {batch_size}, drop_rate {drop_rate}, attention_t {attention_t}, attention_s {attention_s}, heads {heads}.")

    # Check num GPUs
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(f"\nAvailable GPUs: {gpus}\n")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    
    def get_available_devices():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos]

    print(get_available_devices())
    
    # Read data
    save_path = './Multimodal_pretraining/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if data == 'spatial':
        RNA_tf_path, adata_RNA, staining_tf_path = read_data_prepared(data=data, save_path=save_path)
    elif data == 'spatial_split':
        RNA_tf_path, adata_RNA, staining_tf_path, RNA_tf_path_test, adata_RNA_test, staining_tf_path_test = read_data_split_prepared(data=data, save_path=save_path)

    # Train
    weight_path = save_path + 'weight/'
    if train:
        train_concerto(weight_path=weight_path, 
                       RNA_tf_path=RNA_tf_path, 
                       staining_tf_path=staining_tf_path, 
                       data=data, 
                       attention_t=attention_t, 
                       attention_s=attention_s, 
                       batch_size=batch_size, 
                       epoch=epoch, 
                       lr=lr, 
                       drop_rate=drop_rate, 
                       heads=heads, 
                       combine_omics=combine_omics, 
                       model_type=model_type,
                       mask=mask,
                       model_type_image=model_type_image)

    if test:
        if data == "spatial_split":
            res_df = test_concerto_full(adata_RNA=adata_RNA, adata_RNA_test=adata_RNA_test, weight_path=weight_path, data=data, 
                                        RNA_tf_path=RNA_tf_path, staining_tf_path=staining_tf_path_test, 
                                        RNA_tf_path_test=RNA_tf_path_test, staining_tf_path_test=staining_tf_path_test, 
                                        attention_t=attention_t, attention_s=attention_s, mask=mask,
                                        batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
                                        heads=heads, combine_omics=combine_omics, model_type=model_type, cell_labels=cell_labels, model_type_image=model_type_image) 
            filename = f'./Multimodal_pretraining/results/{data}/qr_{mask}_{combine_omics}_mt_{model_type}_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}_{model_type_image}.csv'
            res_df.to_csv(filename)
        
        else:
            # Test on train data
            adata_merged = test_concerto(adata_RNA=adata_RNA, weight_path=weight_path, data=data, 
                                        RNA_tf_path_test=RNA_tf_path, staining_tf_path=staining_tf_path, 
                                        attention_t=attention_t, attention_s=attention_s, mask=mask,
                                        batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
                                        heads=heads, combine_omics=combine_omics, model_type=model_type, 
                                        save_path=save_path, train=True, model_type_image=model_type_image)
            
            filename = f'./Multimodal_pretraining/results/{data}/{data}_{mask}_train_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}_both.h5ad'
            save_merged_adata(adata_merged=adata_merged, filename=filename)

main()
