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

def prepare_data_neurips_cite_together(train: bool = True, save_path: str = ''):
    print("Read human data")
    adata_RNA = sc.read_h5ad(save_path + f'adata_GEX_multiome_train.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_ADT_multiome_train.h5ad')

    adata_RNA_test = sc.read_h5ad(save_path + f'adata_GEX_multiome_test.h5ad')
    adata_Protein_test = sc.read_h5ad(save_path + f'adata_ADT_multiome_test.h5ad')

    print(f"GEX data shape train {adata_RNA.shape}, test {adata_RNA_test.shape}")
    print(f"ADT data shape train {adata_Protein.shape}, test {adata_Protein_test.shape}")

    path_file = 'tfrecord_train/'
    RNA_tf_path = save_path + path_file + 'GEX_multiome_tf/'
    Protein_tf_path = save_path + path_file + 'ADT_multiome_tf/'

    path_file = 'tfrecord_test/'
    RNA_tf_path_test = save_path + path_file + 'GEX_multiome_tf/'
    Protein_tf_path_test = save_path + path_file + 'ADT_multiome_tf/'
    return RNA_tf_path, Protein_tf_path, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_RNA_test

def prepare_data_neurips_multiome_together(train: bool = True, save_path: str = ''):
    print("Read human data")
    adata_RNA = sc.read_h5ad(save_path + f'adata_GEX_multiome_train.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_ADT_multiome_train.h5ad')

    adata_RNA_test = sc.read_h5ad(save_path + f'adata_GEX_multiome_test.h5ad')
    adata_Protein_test = sc.read_h5ad(save_path + f'adata_ADT_multiome_test.h5ad')

    print(f"GEX data shape train {adata_RNA.shape}, test {adata_RNA_test.shape}")
    print(f"ADT data shape train {adata_Protein.shape}, test {adata_Protein_test.shape}")

    path_file = 'tfrecord_train/'
    RNA_tf_path = save_path + path_file + 'GEX_multiome_tf/'
    Protein_tf_path = save_path + path_file + 'ADT_multiome_tf/'

    path_file = 'tfrecord_test/'
    RNA_tf_path_test = save_path + path_file + 'GEX_multiome_tf/'
    Protein_tf_path_test = save_path + path_file + 'ADT_multiome_tf/'
    return RNA_tf_path, Protein_tf_path, adata_RNA, RNA_tf_path_test, Protein_tf_path_test, adata_RNA_test


def read_data(data: str = "human", save_path: str = "", task=0):
    if data == "human":
        GEX_cite_tf_path, ADT_cite_tf_path, adata_GEX_cite, GEX_cite_tf_path_test, ADT_cite_tf_path_test, adata_GEX_cite_test = prepare_data_neurips_cite_together(train=True, save_path=save_path)
        GEX_multiome_tf_path, ADT_multiome_tf_path, adata_GEX_multiome, GEX_multiome_tf_path_test, ADT_multiome_tf_path_test, adata_GEX_multiome_test = prepare_data_neurips_multiome_together(train=True, save_path=save_path)
        return GEX_cite_tf_path, ADT_cite_tf_path, adata_GEX_cite, GEX_cite_tf_path_test, ADT_cite_tf_path_test, adata_GEX_cite_test, GEX_multiome_tf_path, ADT_multiome_tf_path, adata_GEX_multiome, GEX_multiome_tf_path_test, ADT_multiome_tf_path_test, adata_GEX_multiome_test
    else:
        raise Exception("Invalid dataset name.")

def train_cellbind(data, weight_path, 
                   GEX_cite_tf_path, ADT_cite_tf_path, 
                   GEX_multiome_tf_path, ADT_multiome_tf_path,
                   attention_t, attention_s, 
                   batch_size, epoch, lr, drop_rate, 
                   heads, combine_omics, model_type):
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
    else:
        raise Exception("Invalid Teacher/Student combination.")

    print("Trained.")


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
    
    res_df = pd.DataFrame(columns=["accuracy", "f1_median", "f1_macro", "f1_weighted", "pearson" ])
    
    for repeat in range(0, 1):
        GEX_cite_tf_path, ADT_cite_tf_path, adata_GEX_cite, GEX_cite_tf_path_test, ADT_cite_tf_path_test, adata_GEX_cite_test, GEX_multiome_tf_path, ADT_multiome_tf_path, adata_GEX_multiome, GEX_multiome_tf_path_test, ADT_multiome_tf_path_test, adata_GEX_multiome_test = read_data(data=data, save_path=save_path, task=task)

        # Train
        weight_path = save_path + 'weight/'
        if train:
            train_cellbind(data=data, weight_path=weight_path, 
                           GEX_cite_tf_path=GEX_cite_tf_path, ADT_cite_tf_path=ADT_cite_tf_path, 
                           GEX_multiome_tf_path=GEX_multiome_tf_path, ADT_multiome_tf_path=ADT_multiome_tf_path,
                           attention_t=attention_t, attention_s=attention_s, 
                           batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
                           heads=heads, combine_omics=combine_omics, model_type=model_type)
        print("Trained.")

        # if test:    
        #     res_df.loc[repeat] = [acc, f1_median, f1_macro, f1_weighted, pearson]
    # res_df.to_csv(f'./Multimodal_pretraining/results/{data}_qr_train_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.csv')

main()
