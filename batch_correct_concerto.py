import os
import sys
import argparse

import warnings
sys.path.append("../")
import datetime

from concerto_function5_3 import *
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os.path as pth
import pickle # or pickle5
import pandas as pd

import tensorflow as tf
import scipy.sparse as sps

from utils.prepare_dataset import prepare_dataset

import matplotlib.style as style
style.use('ggplot')
sc.settings.set_figure_params(dpi=200, facecolor='white',frameon=False,fontsize=5.5)


def set_gpus():
    os.environ["CUDA_VISIBLE_DEVICES"] = '5' 
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True) 


def get_args():
    parser = argparse.ArgumentParser(description='CONCERTO Batch Correction.')
    parser.add_argument('--epoch', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--lr', type= float, required=True,
                        help='learning rate')
    parser.add_argument('--batch_size', type= int, required=True,
                        help='batch size')
    parser.add_argument('--drop_rate', type= float, required=True,
                        help='dropout rate')
    parser.add_argument('--heads', type= int, required=True,
                        help='dropout rate')
    parser.add_argument('--attention_t', type= bool, required=True,
                        help='dropout rate')
    parser.add_argument('--attention_s', type= bool, required=True,
                        help='dropout rate')

    args = parser.parse_args()
    return args

def prepare_SimulatedConcerto(data_root):
    '''
      return:
         X:         scipy.sparse.csr_matrix, row = feature, column = cell
         gene_name: array of feature (gene) names
         cell_name: array of cell (barcodes) names
         df_meta:   metadata of dataset, columns include 'batchlb'(batch column), 'CellType'(optional)
    '''
    # ===========
    # example

    # label_key = 'final_annotation'

    adata = sc.read(os.path.join(data_root, 'expBatch1_woGroup2.loom'))  # read simulated dataset
    adata = preprocessing_rna(adata, n_top_features=2000, is_hvg=True, batch_key='Batch')
    # X is already sparse after preprocessingRNA function.
    X = adata.X.T # must be gene by cell TODO: true?

    gene_name = adata.var_names.values
    cell_name = adata.obs_names.values
    df_meta = adata.obs[["Batch", "Group"]].copy()

    df_meta["batchlb"] = df_meta["Batch"].astype('category')
    df_meta["CellType"] = df_meta["Group"].astype('category')

    return X, gene_name, cell_name, df_meta

def preprocess_dataset(sps_x, cell_name, gene_name, df_meta, select_hvg=None, scale=False, batch_key="batchlb", label_key="CellType"):
    # compute hvg first, anyway
    adata = sc.AnnData(sps.csr_matrix(sps_x.T))  # transposed, (gene, cell) -> (cell, gene)
    adata.obs_names = cell_name
    adata.var_names = gene_name
    adata.obs = df_meta.loc[cell_name].copy()
    sc.pp.filter_genes(adata, min_cells=3) 
    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)
    if select_hvg is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=min(adata.shape[1], select_hvg), 
                                    # min_mean=0.0125, max_mean=3, min_disp=0.5,
                                    batch_key=batch_key)
        adata = adata[:, adata.var.highly_variable].copy()
    if scale:
        warnings.warn('Scaling per batch! This may cause memory overflow!')
        ada_batches = []
        for bi in adata.obs[batch_key].unique():
            bidx = adata.obs[batch_key] == bi
            adata_batch = adata[bidx].copy()
            sc.pp.scale(adata_batch)
            ada_batches.append(adata_batch)
        adata = sc.concat(ada_batches)
    X = sps.csr_matrix(adata.X)    # some times 
    df_meta = adata.obs.copy()
    cell_name = adata.obs_names
    df_meta[[batch_key, label_key]] = df_meta[[batch_key, label_key]].astype('category')
    return X, cell_name, adata.var_names, df_meta


def main():
    args = get_args()
    epoch = args.epoch
    lr = args.lr
    batch_size= args.batch_size
    drop_rate= args.drop_rate
    attention_t = args.attention_t
    attention_s = args.attention_s
    heads = args.heads
    print(f"Batch correction: epoch {epoch}, lr {lr}, batch_size {batch_size}, drop_rate {drop_rate}, 
          attention_t {attention_t}, attention_s {attention_s}, heads {heads}.")

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True) 
        print(f'GPU info: \n{gpu}')


    path = './Batch_correction/data/expBatch1_woGroup2.loom'
    adata = sc.read(path)
    adata = preprocessing_rna(adata,n_top_features=2000,is_hvg=True,batch_key='Batch')
    print(f"Simulated data shape {adata.shape}")
    print(f"Simulated data: \n {adata}")

    save_path = f'./Batch_correction/data/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    adata.write_h5ad(save_path + 'adata_sim.h5ad')

    print("Making the tfrecord")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sim_tf_path = concerto_make_tfrecord(adata,tf_path = save_path + 'tfrecord/sim_tf/', batch_col_name = 'Batch')
    print("tfrecord length is ok if the two lengths above are ok")

    print("Training Concerto")
    weight_path = save_path + 'weight/'
    sim_tf_path = save_path + 'tfrecord/sim_tf/'
    
    concerto_train_ref(sim_tf_path,weight_path,super_parameters={'batch_size': batch_size, 'epoch': epoch, 'lr': lr, 'drop_rate': drop_rate, 'attention_t': attention_t, 'attention_s': attention_s, 'heads': heads})
    print("Done with training.")

    for dr in [drop_rate, 0.0]:
        for nn in ["encoder", "decoder"]:
            saved_weight_path = save_path + f'weight/weight_{nn}_epoch_{epoch}_{lr}_{drop_rate}.h5'# You can choose a trained weight or use None to default to the weight of the last epoch.
            embedding, sim_id = concerto_test_ref(weight_path,sim_tf_path,super_parameters = {'batch_size': batch_size, 'epoch': 1, 'lr': lr, 'drop_rate': dr, 'attention_t': attention_t, 'heads': heads}, saved_weight_path = saved_weight_path)
            
            # if not os.path.exists(f"{save_path}/embeddings/"):
            #     os.makedirs(f"{save_path}/embeddings/")
            # np.save(f"{save_path}/embeddings/embedding_{nn}_{epoch}_{lr}_{drop_rate}_{dr}_{attention_s}_{attention_t}.csv", embedding)

            print(f'Embedding shape: {embedding.shape}')

            print("Plotting")
            adata = sc.read(f'{save_path}/adata_sim.h5ad')
            adata_1 = adata[sim_id]
            adata_1.obsm['X_embedding'] = embedding

            sc.pp.neighbors(adata_1,n_neighbors=15, use_rep='X_embedding')
            sc.tl.umap(adata_1,min_dist=0.001)

            plt.rcParams.update({
                'svg.fonttype':'none',
                "font.size":5.5,
                'axes.labelsize': 5.5,
                'axes.titlesize':5,
                'legend.fontsize': 5,
                'ytick.labelsize':5,
                'xtick.labelsize':5,
            })

            adata_1.uns['Group_colors'] = ['#ff7f0e','#1f77b4', '#279e68', '#d62728', '#aa40fc', '#8c564b','#e377c2']
            cm = 1/2.54
            fig, axes = plt.subplots(2, 1,figsize=(8*cm,10*cm))
            sc.pl.umap(adata_1, color=['Group'], show=False, ax=axes[0], size=1)
            sc.pl.umap(adata_1, color=['Batch'], show=False, ax=axes[1], size=1)
            fig.tight_layout()

            plt.savefig(f'./Batch_correction/plots/simulated_{nn}_{epoch}_{lr}_{drop_rate}_{dr}_{attention_s}_{attention_t}.png')
