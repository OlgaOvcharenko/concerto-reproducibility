import os
import sys
sys.path.append("../")
from concerto_function5_3 import *
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, silhouette_samples
import tensorflow as tf
import scvelo as scv

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

    args = parser.parse_args()
    return args


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
print(f"Multimodal correction: epoch {epoch}, model type {model_type}, lr {lr}, batch_size {batch_size}, drop_rate {drop_rate}, attention_t {attention_t}, attention_s {attention_s}, heads {heads}.")


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(f"\nAvailable GPUs: {gpus}\n")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


if data == "simulated":
    path = './Multimodal_pretraining/data/multi_gene_l2.loom'
    adata_RNA = sc.read(path)

    path = './Multimodal_pretraining/data/multi_protein_l2.loom'
    adata_Protein = sc.read(path) #cell_type batch

    print("Read simulated data")

    print("Read data.")
    print(f"{data} RNA data shape {adata_RNA.shape}")
    print(f"{data} Protein data shape {adata_Protein.shape}")

    # Create PCA for benchmarking
    adata_merged_tmp = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged_tmp)

    adata_RNA = preprocessing_changed_rna(adata_RNA,min_features = 0, is_hvg=True,batch_key='batch')
    adata_Protein = preprocessing_changed_rna(adata_Protein,min_features = 0, is_hvg=True,batch_key='batch')
    
    # Add PCA after preprocessing for benchmarking
    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]
    adata_merged.obsm["Unintegrated"] = adata_merged_tmp.obsm["X_pca"]
    del adata_merged_tmp
    
    print("Preprocessed data.")

    print(f"{data} RNA data shape {adata_RNA.shape}")
    print(f"{data} Protein data shape {adata_Protein.shape}")
    print(f"{data} RNA data: \n {adata_RNA}")
    print(f"{data} Protein data: \n {adata_Protein}")

    save_path = './Multimodal_pretraining/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # adata_RNA.write_h5ad(save_path + 'adata_RNA.h5ad')
    # adata_Protein.write_h5ad(save_path + 'adata_Protein.h5ad')

    # print("Saved adata.")

    # RNA_tf_path = concerto_make_tfrecord(adata_RNA,tf_path = save_path + 'tfrecord/RNA_tf/',batch_col_name = 'batch')
    # Protein_tf_path = concerto_make_tfrecord(adata_Protein,tf_path = save_path + 'tfrecord/Protein_tf/',batch_col_name = 'batch')
    # print("Make tf record.")

    RNA_tf_path = save_path + 'tfrecord/RNA_tf/'
    Protein_tf_path = save_path + 'tfrecord/Protein_tf/'

else:
    adata_merged_tmp = sc.read_h5ad("./Multimodal_pretraining/data/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad")
    adata_RNA = adata_merged_tmp[:, 0:13431] # adata_gex
    adata_Protein = adata_merged_tmp[:, 13431:] # adata_atac

    print("Read human data")

    # Create PCA for benchmarking
    sc.tl.pca(adata_merged_tmp)

    # FIXME why 20K
    adata_RNA = preprocessing_changed_rna(adata_RNA, min_features = 0, is_hvg=True, batch_key='batch')
    adata_Protein = preprocessing_changed_rna(adata_Protein, min_features = 0, is_hvg=True, batch_key='batch')
    
    # Add PCA after preprocessing for benchmarking
    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]
    adata_merged_tmp.obsm["Unintegrated"] = adata_merged_tmp.obsm["X_pca"]

    del adata_merged_tmp
    
    print("Preprocessed data.")

    print(f"{data} RNA data shape {adata_RNA.shape}")
    print(f"{data} Protein data shape {adata_Protein.shape}")
    print(f"{data} RNA data: \n {adata_RNA}")
    print(f"{data} Protein data: \n {adata_Protein}")

    save_path = './Multimodal_pretraining/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # adata_RNA.write_h5ad(save_path + 'adata_gex.h5ad')
    # adata_Protein.write_h5ad(save_path + 'adata_atac.h5ad')

    # print("Saved adata.")

    # RNA_tf_path = concerto_make_tfrecord(adata_RNA, tf_path = save_path + 'tfrecord/gex_tf/', batch_col_name = 'batch')
    # Protein_tf_path = concerto_make_tfrecord(adata_Protein, tf_path = save_path + 'tfrecord/atac_tf/', batch_col_name = 'batch')
    # print("Make tf record.")

    RNA_tf_path = save_path + 'tfrecord/gex_tf/'
    Protein_tf_path = save_path + 'tfrecord/atac_tf/'

# Train
weight_path = save_path + 'weight/'
if train == 1:
    if attention_t == True and attention_s == False:
        concerto_train_multimodal(['RNA','Protein'] if data == 'simulated' else ['GEX', 'ATAC'],
                                RNA_tf_path,Protein_tf_path,weight_path, 
                                super_parameters={
                                    'data': data,
                                    'batch_size': batch_size, 
                                    'epoch_pretrain': epoch, 'lr': lr, 
                                    'drop_rate': drop_rate, 
                                    'attention_t': attention_t, 
                                    'attention_s': attention_s, 
                                    'heads': heads,
                                    'combine_omics': False,
                                    'model_type': model_type
                                    })
    elif attention_t == True and attention_s == True:
        concerto_train_multimodal_tt(['RNA','Protein'] if data == 'simulated' else ['GEX', 'ATAC'],
                                RNA_tf_path,Protein_tf_path,weight_path, 
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
    elif attention_t == False and attention_s == False:
        concerto_train_multimodal_ss(['RNA','Protein'] if data == 'simulated' else ['GEX', 'ATAC'],
                                RNA_tf_path,Protein_tf_path,weight_path, 
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

ep_vals = []
i = 4
while i < epoch:
    ep_vals.append(i)
    i = i * 2
ep_vals.append(epoch)

adata_merged.obs = adata_RNA.obs

# Test
if test:
    diverse_tests_names = []
    for dr in [drop_rate]:
        for nn in ["encoder"]:
            for e in ep_vals: 
                saved_weight_path = f'./Multimodal_pretraining/weight/multi_weight_{nn}_{data}_{batch_size}_model_{model_type}_epoch_{e}_{lr}_{drop_rate}_{attention_t}_{attention_s}_{heads}.h5'
                # saved_weight_path = f'./Multimodal_pretraining/weight/multi_weight_{nn}_{data}_model_{model_type}_epoch_{e}_{lr}_{drop_rate}_{attention_t}_{attention_s}_{heads}.h5' # You can choose a trained weight or use None to default to the weight of the last epoch.
                
                if (nn == "decoder" and attention_s == False) or (nn == "encoder" and attention_t == False):
                    embedding,batch, RNA_id, attention_weight =  concerto_test_multimodal_decoder(
                    ['RNA','Protein'] if data == 'simulated' else ['GEX', 'ATAC'],
                    weight_path, 
                    RNA_tf_path,
                    Protein_tf_path,
                    n_cells_for_sample=None,
                    super_parameters={
                        'batch_size': batch_size, 
                        'epoch': e, 'lr': lr, 
                        'drop_rate': dr, 
                        'attention_t': attention_t, 
                        'attention_s': attention_s, 
                        'heads': heads,
                        'combine_omics': False
                    }, 
                    saved_weight_path = saved_weight_path)
                else:
                    embedding, batch, RNA_id, attention_weight =  concerto_test_multimodal(
                        ['RNA','Protein'] if data == 'simulated' else ['GEX', 'ATAC'],
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
                            'combine_omics': False,
                            'model_type': model_type
                        }, 
                        saved_weight_path = saved_weight_path)

                print("Tested.")
                
                if data == "simulated":
                    adata_RNA = sc.read(save_path + 'adata_RNA.h5ad')
                else:
                    adata_RNA = sc.read(save_path + 'adata_gex.h5ad')
                
                # FIXME
                adata_RNA_1 = adata_RNA[RNA_id]
                adata_RNA_1.obsm['X_embedding'] = embedding

                print(f"Shape of the embedding {embedding.shape}.")

                # Add for the later benchmarking 
                adata_merged = adata_merged[RNA_id]

                adata_merged.obsm[f"{e}_{nn}_{dr}"] = embedding
                diverse_tests_names.append(f"{e}_{nn}_{dr}")

                l2tol1 = {
                    'CD8 Naive': 'CD8 T',
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

                sc.pp.neighbors(adata_RNA_1, use_rep="X_embedding", metric="cosine")
                labels = adata_RNA_1.obs['cell_type_l1'].tolist()
                for res in [0.05,0.1,0.15,0.2,0.25,0.3]:
                    sc.tl.leiden(adata_RNA_1, resolution=res)
                    target_preds = adata_RNA_1.obs['leiden'].tolist()
                    nmi = np.round(normalized_mutual_info_score(labels, target_preds), 5)
                    ari = np.round(adjusted_rand_score(labels, target_preds), 5)    
                    n_cluster = len(list(set(target_preds)))
                    print('leiden(res=%f): ari = %.5f , nmi = %.5f, n_cluster = %d' % (res, ari, nmi, n_cluster), '.')

                # sc.pp.neighbors(adata_RNA_1, use_rep='X_embedding', metric='cosine')
                sc.tl.leiden(adata_RNA_1, resolution=0.2)
                sc.tl.umap(adata_RNA_1,min_dist=0.1)
                sc.set_figure_params(dpi=150)
                sc.pl.umap(adata_RNA_1, color=['cell_type_l1','leiden'], legend_fontsize ='xx-small', size=5, legend_fontweight='light')
                plt.savefig(f'./Multimodal_pretraining/plots/{data}/{data}_mt_{model_type}_bs_{batch_size}_{nn}_{e}_{lr}_{drop_rate}_{dr}_{attention_s}_{attention_t}_{heads}.png')
                
                # scv.pl.velocity_embedding(f'./Multimodal_pretraining/plots/{data}/{data}_mt_{model_type}_bs_{batch_size}_{nn}_{e}_{lr}_{drop_rate}_{dr}_{attention_s}_{attention_t}_{heads}.png', basis="umap")

    filename = f'./Multimodal_pretraining/data/full_{data}/{data}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'
    adata_merged.write(filename)

    # Benchmark
    print(adata_merged)
    print(f"Saved adata all at {filename}")
