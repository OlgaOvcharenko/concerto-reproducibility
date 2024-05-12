import os
import sys
sys.path.append("../")
from concerto_function5_3 import *
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, silhouette_samples
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(f"\nAvailable GPUs: {gpus}\n")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

path = './Multimodal_pretraining/data/multi_gene_l2.loom'
adata_RNA = sc.read(path)
path = './Multimodal_pretraining/data/multi_protein_l2.loom'
adata_Protein = sc.read(path) #cell_type batch

print("Read data.")

adata_RNA = preprocessing_rna(adata_RNA,min_features = 0,is_hvg=False,batch_key='batch')
adata_Protein = preprocessing_rna(adata_Protein,min_features = 0,is_hvg=False,batch_key='batch')

print("Preprocessed data.")

save_path = './Multimodal_pretraining/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
adata_RNA.write_h5ad(save_path + 'adata_RNA.h5ad')
adata_Protein.write_h5ad(save_path + 'adata_Protein.h5ad')

print("Saved adata.")

RNA_tf_path = concerto_make_tfrecord(adata_RNA,tf_path = save_path + 'tfrecord/RNA_tf/',batch_col_name = 'batch')
Protein_tf_path = concerto_make_tfrecord(adata_Protein,tf_path = save_path + 'tfrecord/Protein_tf/',batch_col_name = 'batch')

# Train
weight_path = save_path + 'weight/'
RNA_tf_path = save_path + 'tfrecord/RNA_tf/'
Protein_tf_path = save_path + 'tfrecord/Protein_tf/'
concerto_train_multimodal(RNA_tf_path,Protein_tf_path,weight_path,super_parameters={'batch_size': 64, 'epoch_pretrain': 5, 'lr': 1e-4,'drop_rate': 0.1})

print("Trained.")

# Test
saved_weight_path = './Multimodal_pretraining/weight/weight_encoder_epoch3.h5' # You can choose a trained weight or use None to default to the weight of the last epoch.
embedding,batch,RNA_id,attention_weight =  concerto_test_multimodal(weight_path,RNA_tf_path,Protein_tf_path,n_cells_for_sample = None,super_parameters={'batch_size': 32, 'epoch_pretrain': 1, 'lr': 1e-4,'drop_rate': 0.1},saved_weight_path = saved_weight_path)

print("Tested.")

save_path = './'
adata_RNA = sc.read(save_path + 'adata_RNA.h5ad')
adata_RNA_1 = adata_RNA[RNA_id]
adata_RNA_1.obsm['X_embedding'] = embedding

l2tol1 = {'CD8 Naive': 'CD8 T',
 'CD8 Proliferating': 'CD8 T',
 'CD8 TCM': 'CD8 T',
 'CD8 TEM': 'CD8 T',
 'CD4 CTL': 'CD4 T',
 'CD4 Naive': 'CD4 T',
 'CD4 Proliferating': 'CD4 T',
 'CD4 TCM': 'CD4 T',
 'CD4 TEM': 'CD4 T',
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
 'Plasmablast': 'B',
 'Eryth': 'other',
 'HSPC': 'other',
 'Platelet': 'other'
}
adata_RNA_1.obs['cell_type_l1'] = adata_RNA_1.obs['cell_type'].map(l2tol1)
print(adata_RNA_1)

sc.pp.neighbors(adata_RNA_1, use_rep="X_embedding")
labels = adata_RNA_1.obs['cell_type_l1'].tolist()
for res in [0.05,0.1,0.15,0.2,0.25,0.3]:
    sc.tl.leiden(adata_RNA_1, resolution=res)
    target_preds = adata_RNA_1.obs['leiden'].tolist()
    nmi = np.round(normalized_mutual_info_score(labels, target_preds), 5)
    ari = np.round(adjusted_rand_score(labels, target_preds), 5)    
    n_cluster = len(list(set(target_preds)))
    print('leiden(res=%f): ari = %.5f , nmi = %.5f, n_cluster = %d' % (res, ari, nmi, n_cluster), '.')

#sc.pp.neighbors(adata_RNA_1, use_rep='X_embedding')
sc.tl.leiden(adata_RNA_1, resolution=0.2)
sc.tl.umap(adata_RNA_1,min_dist=0.1)
sc.set_figure_params(dpi=150)
sc.pl.umap(adata_RNA_1, color=['cell_type_l1','leiden'],legend_fontsize ='xx-small',size=5,legend_fontweight='light')
