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
    parser = argparse.ArgumentParser(description='Evaluation Script for scRNA-seq Contrastive Learning Models')
    # environ settings
    """parser.add_argument('--path', default='/cluster/work/boeva/tomap', type=str,
                        help='Path to where the embeddings can be found')
    parser.add_argument('--model-paths', nargs='+', default=["CLAIRE-outputs"],
                        help='Path to the embeddings of the models to be benchmarked')"""
    parser.add_argument('--dname', default='Simulated', type=str,
                        help='Name of the dataset used for benchmarking')
    parser.add_argument('--train', action='store_true',
                        help='If --train is not set, only perform inference of the embeddings.')

    args = parser.parse_args()
    return args

def py_read_data(_dir, fname):
    # read data in sps
    # saved in (cells, genes)
    sps_X = sps.load_npz(pth.join(_dir, fname+'.npz'))
    # read gene names
    with open(pth.join(_dir, fname+'_genes.pkl'), 'rb') as f:
        genes = pickle.load(f)
    # read cell names
    with open(pth.join(_dir, fname+'_cells.pkl'), 'rb') as f:
        cells = pickle.load(f)
    return sps_X, cells, genes

def load_meta_txt(path, delimiter='\t'):
    st = datetime.datetime.now()
    data, colname, cname = [], [], []
    with open(path, 'r') as f:
        for li, line in enumerate(f):
            line = line.strip().replace("\"", '').split(delimiter)
            if li==0:
                colname = line
                continue
            cname.append(line[0])
            data.append(line[1:])
    df = pd.DataFrame(data, columns=colname, index=cname)
    ed = datetime.datetime.now()
    total_seconds = (ed-st).total_seconds() * 1.0
    print('The reading cost time {:.4f} secs'.format(total_seconds))
    return df

def prepare_PBMC(data_root, batch_key, label_key):
    sps_x1, gene_name1, cell_name1 = py_read_data(data_root, 'b1_exprs')
    sps_x2, gene_name2, cell_name2 = py_read_data(data_root, 'b2_exprs')
    sps_x = sps.hstack([sps_x1, sps_x2])
    cell_name = np.hstack((cell_name1, cell_name2))
    assert np.all(gene_name1 == gene_name2), 'gene order not match'
    gene_name = gene_name1
    df_meta1 = load_meta_txt(pth.join(data_root, 'b1_celltype.txt'))
    df_meta2 = load_meta_txt(pth.join(data_root, 'b2_celltype.txt'))
    df_meta1['batchlb'] = 'Batch1'
    df_meta2['batchlb'] = 'Batch2'
    df_meta = pd.concat([df_meta1, df_meta2])
    df_meta[batch_key] = df_meta[batch_key].astype('category')
    df_meta[label_key] = df_meta[label_key].astype('category')
    return sps_x, gene_name, cell_name, df_meta

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


## ================================== ##
##    Main Function (can be a todo)   ##
## ================================== ##
args = get_args()

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) 
    print(f'GPU info: \n{gpu}')


path = './Batch_correction/data/expBatch1_woGroup2.loom'
dname = args.dname

n_hvgs = 2000
scale = False
batch_key = 'batchlb'
label_key = 'CellType'

# dataset_dir = pth.join(path, "CLAIRE-data", dname)
# """if dname == "Simulated":
#     sps_x, genes, cells, df_meta = prepare_SimulatedConcerto(dataset_dir)    
# else:
#     sps_x, genes, cells, df_meta = prepare_PBMC(dataset_dir, "batchlb", "CellType")"""

# # The function will call the corresponding preparation function for the dataset.
# sps_x, genes, cells, df_meta = prepare_dataset(data_dir=dataset_dir)

# adata, X, cell_name, gene_name, df_meta = preprocess_dataset(
#     sps_x,
#     cells, 
#     genes, 
#     df_meta, 
#     n_hvgs, 
#     scale, 
# )

# adata = sc.AnnData(X)
# adata.var_names = gene_name
# adata.obs = df_meta.copy()

adata = sc.read(path)
adata = preprocessing_rna(adata,n_top_features=2000,is_hvg=True,batch_key='Batch')
print(adata)

# ======================
# Training of the model
# ======================

save_path = f'./Batch_correction/data/{dname}/'
if not os.path.exists(save_path):
    print(f"Dataset {dname} has not been trained on.\nCreate corresponding directory.\n")
    os.makedirs(save_path)
    #os.system(f"mkdir {save_path}")
os.system(f"rm {save_path}/tfrecord/sim_tf/*")

if not os.path.exists(save_path):
    os.makedirs(save_path)
adata.write_h5ad(save_path + 'adata_sim.h5ad')

if not os.path.exists(os.path.join(save_path, "embeddings")):
    os.makedirs(os.path.join(save_path, "embeddings"))

print("Debugging the tfrecord")
if not os.path.exists(save_path):
    os.makedirs(save_path)
sim_tf_path = concerto_make_tfrecord(adata,tf_path = save_path + 'tfrecord/sim_tf/',batch_col_name = batch_key)
print("tfrecord length is ok if the two lengths above are ok.")

print("Training Concerto")
if args.train:
    weight_path = save_path + 'weight/'
    sim_tf_path = save_path + 'tfrecord/sim_tf/'
    concerto_train_ref(sim_tf_path, weight_path, super_parameters={'batch_size': 64, 'epoch': 400, 'lr': 1e-6})
    print("Done with training.")


weight_path = save_path + 'weight/'
sim_tf_path = save_path + 'tfrecord/sim_tf/'

saved_weight_path = save_path + f'weight/weight_encoder_epoch.h5' #You can choose a trained weight or use None to default to the weight of the last epoch.
embedding,sim_id = concerto_test_ref(weight_path,sim_tf_path, 
                                     super_parameters = {
                                         'batch_size': 64, 
                                         'epoch': 10, 
                                         'lr': 1e-5,
                                         'drop_rate': 0.1
                                        }, 
                                     saved_weight_path = saved_weight_path)
np.save(f"{save_path}/embeddings/embedding")

print(f'Embedding shape: {embedding.shape}')

#adata = sc.AnnData(X)
#adata.var_names = gene_name
#adata.obs = df_meta.copy()

# ======================
# Done with training.
# ----------------------
# Continue with plotting
# ======================

print("Plotting")
adata = sc.read(f'{save_path}/adata_sim.h5ad')
adata_1 = adata[sim_id]
adata_1.obsm['X_embedding'] = embedding

sc.pp.neighbors(adata_1,n_neighbors=15, use_rep='X_embedding')
sc.tl.umap(adata_1,min_dist =0.001)

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

plt.savefig(f'{save_path}/test_output_10epochs.png')
