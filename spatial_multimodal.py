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
# from tensorflow.python.client import device_lib

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

def train_concerto(weight_path: str, RNA_tf_path: str, staining_tf_path: str, data: str, 
                   attention_t: bool, attention_s: bool,
                   batch_size:int, epoch: int, lr: float, drop_rate: float, 
                   heads: int, combine_omics: int, model_type: int):
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
                                          'model_type': model_type
                                          })

    print("Trained.")

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

    print(f"Multimodal correction: epoch {epoch}, model type {model_type}, lr {lr}, batch_size {batch_size}, drop_rate {drop_rate}, attention_t {attention_t}, attention_s {attention_s}, heads {heads}.")

    # Check num GPUs
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(f"\nAvailable GPUs: {gpus}\n")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    exit()
    # def get_available_devices():
    #     local_device_protos = device_lib.list_local_devices()
    #     return [x.name for x in local_device_protos]

    # print(get_available_devices())

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
                       model_type=model_type)

main()
