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

    args = parser.parse_args()
    return args

def train_concerto(weight_path: str, RNA_tf_path: str, staining_tf_path: str, data: str, 
                   attention_t: bool, attention_s: bool,
                   batch_size:int, epoch: int, lr: float, drop_rate: float, 
                   heads: int, combine_omics: int, model_type: int, mask: bool):
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
                                          'mask': mask
                                          })

    print("Trained.")

def test_concerto(adata_RNA, weight_path: str, data: str, 
                  RNA_tf_path_test: str, staining_tf_path: str, 
                  attention_t: bool, attention_s: bool,
                  batch_size:int, epoch: int, lr: float, drop_rate: float, 
                  heads: int, combine_omics: int, model_type: int, mask: int,
                  save_path: str, train: bool = False, adata_RNA_train = None):
    ep_vals = []
    i = 4
    while i < epoch:
        ep_vals.append(i)
        i = i * 2
    ep_vals.append(epoch)

    adata_merged = adata_RNA

    # Test
    nn = "encoder"
    dr = 0.0 # drop_rate
    only_images = [True, False] # if combine_omics == 0 else [False]
    for only_image in only_images:
        for e in ep_vals: 
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
                        'mask': mask
                    }, 
                    saved_weight_path = saved_weight_path,
                    only_image=only_image)
            
            print(adata_RNA)

            adata_RNA_1 = adata_RNA[RNA_id]
            adata_RNA_1.obsm['X_embedding'] = embedding

            print(f"\nShape of the {train}_{e}_{nn}_{dr}_{only_image} embedding {embedding.shape}.")
            
            adata_merged = adata_merged[RNA_id]
            adata_merged.obsm[f'train_{e}_{nn}_{dr}_{only_image}' if train else f'test_{e}_{nn}_{dr}_{only_image}'] = embedding
            
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
                color=['cell_type', f'pred_cell_type_{e}_{nn}_{dr}_{only_image}', 'leiden', 'batch']
            else:
                color=['cell_type', 'leiden', 'batch']

            sc.set_figure_params(dpi=150)
            sc.pl.umap(adata_RNA_1, color=color, legend_fontsize ='xx-small', size=5, legend_fontweight='light') # edges=True
            plt.savefig(f'./Multimodal_pretraining/plots/{data}/{data}_{mask}_{"train" if train else "test"}_{combine_omics}_oRNA{only_image}_mt_{model_type}_bs_{batch_size}_{nn}_{e}_{lr}_{drop_rate}_{dr}_{attention_s}_{attention_t}_{heads}.png')

    return adata_merged

def read_data_prepared(data: str = "", save_path: str = ""):
    if data != 'spatial':
        raise Exception('[SPATIAL] Incorrect dataset name.')
    
    print("Read data")
    print("Read spatial data.")
    adata_RNA = sc.read_h5ad(save_path + f'spatial_adata_RNA.h5ad')

    path_file = 'tfrecord/'
    RNA_tf_path = save_path + path_file + 'spatial_RNA_tf/'
    staining_tf_path = save_path + path_file + 'spatial_staining_tf/'
    
    return RNA_tf_path, adata_RNA, staining_tf_path

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

    RNA_tf_path, adata_RNA, staining_tf_path = read_data_prepared(data=data, save_path=save_path)

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
                       mask=mask)

    if test:
        # Test on train data
        adata_merged = test_concerto(adata_RNA=adata_RNA, weight_path=weight_path, data=data, 
                                     RNA_tf_path_test=RNA_tf_path, staining_tf_path=staining_tf_path, 
                                     attention_t=attention_t, attention_s=attention_s, mask=mask,
                                     batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
                                     heads=heads, combine_omics=combine_omics, model_type=model_type, 
                                     save_path=save_path, train=True)
        
        filename = f'./Multimodal_pretraining/data/{data}/{data}_{mask}_train_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'
        save_merged_adata(adata_merged=adata_merged, filename=filename)

        # # Test on test data
        # adata_merged_test = test_concerto(adata_RNA=adata_RNA, weight_path=weight_path, data=data, 
        #                                   RNA_tf_path_test=RNA_tf_path, staining_tf_path=staining_tf_path, 
        #                                   attention_t=attention_t, attention_s=attention_s, mask=mask,
        #                                   batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
        #                                   heads=heads, combine_omics=combine_omics, model_type=model_type, 
        #                                   save_path=save_path, train=False, adata_RNA_train=adata_merged)

        # filename = f'./Multimodal_pretraining/data/{data}/{data}_{mask}_test_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'
        # save_merged_adata(adata_merged=adata_merged_test, filename=filename)

main()
