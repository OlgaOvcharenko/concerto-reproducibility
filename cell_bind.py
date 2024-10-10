import os
import sys

import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
sys.path.append("../")
from cellbind_function import *
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import tensorflow as tf
from statistics import mode

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
    parser.add_argument('--batch_size2', type= int, required=True,
                        help='batch size2')
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
    adata_RNA = sc.read_h5ad(save_path + f'adata_cellbind_GEX_train.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_cellbind_ADT_train.h5ad')

    adata_RNA_test = sc.read_h5ad(save_path + f'adata_cellbind_GEX_test.h5ad')
    adata_Protein_test = sc.read_h5ad(save_path + f'adata_cellbind_ADT_test.h5ad')

    print(f"GEX data shape train {adata_RNA.shape}, test {adata_RNA_test.shape}")
    print(f"ADT data shape train {adata_Protein.shape}, test {adata_Protein_test.shape}")

    path_file = 'tfrecord_train/'
    RNA_tf_path = save_path + path_file + 'GEX_cellbind_tf/'
    Protein_tf_path = save_path + path_file + 'ADT_cellbind_tf/'

    path_file = 'tfrecord_test/'
    RNA_tf_path_test = save_path + path_file + 'GEX_cellbind_tf/'
    Protein_tf_path_test = save_path + path_file + 'ADT_cellbind_tf/'
    return RNA_tf_path, Protein_tf_path, adata_RNA, adata_Protein, RNA_tf_path_test, Protein_tf_path_test, adata_RNA_test, adata_Protein_test

def prepare_data_neurips_multiome_together(train: bool = True, save_path: str = ''):
    print("Read human data")
    adata_RNA = sc.read_h5ad(save_path + f'adata_cellbind_GEX_multiome_train.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_cellbind_ATAC_multiome_train.h5ad')

    adata_RNA_test = sc.read_h5ad(save_path + f'adata_cellbind_GEX_multiome_test.h5ad')
    adata_Protein_test = sc.read_h5ad(save_path + f'adata_cellbind_ATAC_multiome_test.h5ad')

    print(f"GEX data shape train {adata_RNA.shape}, test {adata_RNA_test.shape}")
    print(f"ADT data shape train {adata_Protein.shape}, test {adata_Protein_test.shape}")

    path_file = 'tfrecord_train/'
    RNA_tf_path = save_path + path_file + 'GEX_cellbind_multiome_tf/'
    Protein_tf_path = save_path + path_file + 'ATAC_cellbind_multiome_tf/'

    path_file = 'tfrecord_test/'
    RNA_tf_path_test = save_path + path_file + 'GEX_cellbind_multiome_tf/'
    Protein_tf_path_test = save_path + path_file + 'ATAC_cellbind_multiome_tf/'
    return RNA_tf_path, Protein_tf_path, adata_RNA, adata_Protein, RNA_tf_path_test, Protein_tf_path_test, adata_RNA_test, adata_Protein_test


def read_data(data: str = "human", save_path: str = "", task=0):
    if data == "human":
        GEX_cite_tf_path, ADT_cite_tf_path, adata_GEX_cite, adata_ADT_cite, GEX_cite_tf_path_test, ADT_cite_tf_path_test, adata_GEX_cite_test, adata_ADT_cite_test = prepare_data_neurips_cite_together(train=True, save_path=save_path)
        GEX_multiome_tf_path, ATAC_multiome_tf_path, adata_GEX_multiome, adata_ATAC_multiome, GEX_multiome_tf_path_test, ATAC_multiome_tf_path_test, adata_GEX_multiome_test, adata_ATAC_multiome_test = prepare_data_neurips_multiome_together(train=True, save_path=save_path)
        return GEX_cite_tf_path, ADT_cite_tf_path, adata_GEX_cite, adata_ADT_cite, GEX_cite_tf_path_test, ADT_cite_tf_path_test, adata_GEX_cite_test, adata_ADT_cite_test, GEX_multiome_tf_path, ATAC_multiome_tf_path, adata_GEX_multiome, adata_ATAC_multiome, GEX_multiome_tf_path_test, ATAC_multiome_tf_path_test, adata_GEX_multiome_test, adata_ATAC_multiome_test
    else:
        raise Exception("Invalid dataset name.")

def train_cellbind(data, weight_path, 
                   GEX_cite_tf_path, ADT_cite_tf_path, 
                   GEX_multiome_tf_path, ATAC_multiome_tf_path,
                   attention_t, attention_s, 
                   batch_size, batch_size2, epoch, lr, drop_rate, 
                   heads, combine_omics, model_type):
    if attention_t == True and attention_s == False:
        super_parameters = {'data': data,
                           'batch_size12': batch_size,
                           'batch_size13': batch_size2,
                           'epoch_pretrain': epoch, 
                           'lr': lr, 
                           'drop_rate': drop_rate, 
                           'attention_t': attention_t, 
                           'attention_s': attention_s, 
                           'heads': heads,
                           'combine_omics': combine_omics,
                           'model_type': model_type
                           }
        
        GEX_network = create_single_cell_network(mult_feature_name='GEX', 
                                                 tf_path=GEX_cite_tf_path, 
                                                 super_parameters=super_parameters)
        ADT_network = create_single_cell_network(mult_feature_name='ADT', 
                                                 tf_path=ADT_cite_tf_path, 
                                                 super_parameters=super_parameters)
        ATAC_network = create_single_cell_network(mult_feature_name='ATAC', 
                                                 tf_path=ATAC_multiome_tf_path, 
                                                 super_parameters=super_parameters)
        cellbind_train_multimodal(mod1a_tf_path=GEX_cite_tf_path, 
                                  mod2_tf_path=ADT_cite_tf_path, 
                                  mod1b_tf_path=GEX_multiome_tf_path, 
                                  mod3_tf_path=ATAC_multiome_tf_path,
                                  weight_path=weight_path, 
                                  mod1_network=GEX_network, 
                                  mod2_network=ADT_network, 
                                  mod3_network=ATAC_network,
                                  super_parameters=super_parameters)
    else:
        raise Exception("Invalid Teacher/Student combination.")

    print("Trained.")

def qr_test(adata_ref, embedding_ref, embedding_query, test_cell_types):
    query_neighbor, _ = knn_classifier(ref_embedding=embedding_ref, query_embedding=embedding_query, ref_anndata=adata_ref, column_name='cell_type_l1', k=5)
    cell_types_list = pd.unique(test_cell_types).tolist() 
    acc = accuracy_score(test_cell_types.to_list(), query_neighbor)
    f1 = f1_score(test_cell_types.to_list(), query_neighbor, labels=cell_types_list, average=None)
    f1_weighted = f1_score(test_cell_types.to_list(), query_neighbor, labels=cell_types_list, average='weighted')
    f1_macro = f1_score(test_cell_types.to_list(), query_neighbor, labels=cell_types_list, average='macro')
    f1_median = np.median(f1)
    
    print(f"\nQR")
    print(f"Per class {cell_types_list} F1 {f1}")
    print('Accuracy {:.3f}, F1 median {:.3f}, F1 macro {:.3f}, F1 weighted {:.3f} '.format(acc, f1_median, f1_macro, f1_weighted),)

    res_dict = {"Accuracy": acc, "F1 median": f1_median, "F1 macro": f1_macro, "F1 weigted": f1_weighted}
    for k, v in zip(cell_types_list, f1):
        res_dict[k] = v
    return res_dict

def mp_3mod_test(embedding_train, embedding_test, adata_unknown_train, adata_unknown_test):
    nbrs = NearestNeighbors(metric='cosine', n_neighbors=5, algorithm='auto').fit(embedding_train)
    indices = nbrs.kneighbors(embedding_test, return_distance=False)

    val_new_protein = np.array(adata_unknown_train.X.todense())[indices].mean(axis=1)
    tmp = adata_unknown_test.X.todense()

    pearsons = []
    for true_protein, pred_protein in zip(tmp, val_new_protein):
        pearsons.append(np.corrcoef(pred_protein, true_protein)[0, 1])

    print(f'\nKnown modality Pearson: {np.mean(pearsons)}')
    return np.mean(pearsons)

def mp_2mod_unknown_test(embedding_train, embedding_test, adata_unknown_train, adata_unknown_test):
    nbrs = NearestNeighbors(metric='cosine', n_neighbors=5, algorithm='auto').fit(embedding_train)
    indices = nbrs.kneighbors(embedding_test, return_distance=False)

    # FIXME remove
    tmp = np.array(adata_unknown_train.obs["cell_type_l1"])[indices]
    # print(tmp)
    tmp_res = []
    for val in tmp:
        g = mode(val)
        # print(g)
        tmp_res.append(g)
    # print(np.array(tmp_res))

    # np.array(adata_unknown_train.obs["cell_type_l1"])[indices].mode()
    val_new_unknown = np.array(tmp_res) 
    
    test_cell_types = adata_unknown_test.obs["cell_type_l1"]
    cell_types_list = pd.unique(test_cell_types).tolist() 
    acc = accuracy_score(test_cell_types.to_list(), val_new_unknown)
    f1 = f1_score(test_cell_types.to_list(), val_new_unknown, labels=cell_types_list, average=None)
    f1_weighted = f1_score(test_cell_types.to_list(), val_new_unknown, labels=cell_types_list, average='weighted')
    f1_macro = f1_score(test_cell_types.to_list(), val_new_unknown, labels=cell_types_list, average='macro')
    f1_median = np.median(f1)
    
    print(f"\nUnknown modality 2")
    print(f"Per class {cell_types_list} F1 {f1}")
    print('Accuracy {:.3f}, F1 median {:.3f}, F1 macro {:.3f}, F1 weighted {:.3f} '.format(acc, f1_median, f1_macro, f1_weighted),)

    res_dict = {"Accuracy": acc, "F1 median": f1_median, "F1 macro": f1_macro, "F1 weigted": f1_weighted}
    for k, v in zip(cell_types_list, f1):
        res_dict[k] = v

    # Correlation between two GEX averaged and true
    nbrs = NearestNeighbors(metric='cosine', n_neighbors=5, algorithm='auto').fit(embedding_train)
    indices = nbrs.kneighbors(embedding_test, return_distance=False)
    # print(np.array(adata_unknown_train.X.todense())[indices].shape)
    val_new_GEX = np.array(adata_unknown_train.X.todense())[indices].mean(axis=1)
    tmp_a = adata_unknown_test.X.todense()
    # print(val_new_GEX.shape)
    # print(tmp_a.shape)

    pearsons = []
    for true_protein, pred_protein in zip(tmp_a, val_new_GEX):
        # print(true_protein.shape)
        # print(pred_protein.shape)
        # print(np.corrcoef(pred_protein, true_protein))
        pearsons.append(np.corrcoef(pred_protein, true_protein)[0, 1])

    res_dict["pearsons"] = np.mean(pearsons)
    print(f'\GEX Pearson: {np.mean(pearsons)}')

    return res_dict

def test_cellbind(adata_cite_GEX, adata_cite_GEX_test,
                  adata_ADT, adata_ADT_test,
                  adata_multiome_GEX, adata_multiome_GEX_test,
                  adata_ATAC, adata_ATAC_test,
                  cite_GEX_tf_path: str, multiome_GEX_tf_path: str, ADT_tf_path: str, ATAC_tf_path: str, 
                  cite_GEX_tf_path_test: str, multiome_GEX_tf_path_test: str, ADT_tf_path_test: str, ATAC_tf_path_test: str, 
                  weight_path: str,
                  data: str, attention_t: bool, attention_s: bool, batch_size:int, batch_size2:int, epoch: int, lr: float, drop_rate: float, 
                  heads: int, combine_omics: int, model_type: int):
    
    super_parameters = {'data': data,
                        'batch_size12': batch_size,
                        'batch_size13': batch_size2,
                        'epoch_pretrain': epoch, 
                        'lr': lr, 
                        'drop_rate': drop_rate, 
                        'attention_t': attention_t, 
                        'attention_s': attention_s, 
                        'heads': heads,
                        'combine_omics': combine_omics,
                        'model_type': model_type
                        }
        
    GEX_network = create_single_cell_network(mult_feature_name='GEX', 
                                                tf_path=cite_GEX_tf_path_test, 
                                                super_parameters=super_parameters)
    ADT_network = create_single_cell_network(mult_feature_name='ADT', 
                                                tf_path=ADT_tf_path_test, 
                                                super_parameters=super_parameters)
    ATAC_network = create_single_cell_network(mult_feature_name='ATAC', 
                                                tf_path=ATAC_tf_path_test, 
                                                super_parameters=super_parameters)
    saved_weight_path_GEX = weight_path + f'GEX_weight_{super_parameters["data"]}_{super_parameters["batch_size12"]}_{super_parameters["batch_size13"]}_model_{super_parameters["combine_omics"]}_{super_parameters["model_type"]}_epoch_{epoch}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_t"]}_{super_parameters["attention_s"]}_{super_parameters["heads"]}.h5'
    saved_weight_path_ADT = weight_path + f'ADT_weight_{super_parameters["data"]}_{super_parameters["batch_size12"]}_{super_parameters["batch_size13"]}_model_{super_parameters["combine_omics"]}_{super_parameters["model_type"]}_epoch_{epoch}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_t"]}_{super_parameters["attention_s"]}_{super_parameters["heads"]}.h5'
    saved_weight_path_ATAC = weight_path + f'ATAC_weight_{super_parameters["data"]}_{super_parameters["batch_size12"]}_{super_parameters["batch_size13"]}_model_{super_parameters["combine_omics"]}_{super_parameters["model_type"]}_epoch_{epoch}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_t"]}_{super_parameters["attention_s"]}_{super_parameters["heads"]}.h5'


    result_dicts = []
    for concat_mod in [False, True]:
        res12_train = cellbind_test_multimodal(mod1_tf_path=cite_GEX_tf_path, mod2_tf_path=ADT_tf_path,
                                mod1_network=GEX_network, mod2_network=ADT_network, batch_size=batch_size,
                                saved_weight_path=saved_weight_path_GEX, saved_weight_path2=saved_weight_path_ADT, 
                                super_parameters=super_parameters, concat_modalities=concat_mod)
        
        res12_test = cellbind_test_multimodal(mod1_tf_path=cite_GEX_tf_path_test, mod2_tf_path=ADT_tf_path_test,
                                mod1_network=GEX_network, mod2_network=ADT_network, batch_size=batch_size,
                                saved_weight_path=saved_weight_path_GEX, saved_weight_path2=saved_weight_path_ADT, 
                                super_parameters=super_parameters, concat_modalities=concat_mod)
        
        res13_train = cellbind_test_multimodal(mod1_tf_path=multiome_GEX_tf_path, mod2_tf_path=ATAC_tf_path,
                                mod1_network=GEX_network, mod2_network=ATAC_network, batch_size=batch_size2,
                                saved_weight_path=saved_weight_path_GEX, saved_weight_path2=saved_weight_path_ATAC, 
                                super_parameters=super_parameters, concat_modalities=concat_mod)
        
        res13_test = cellbind_test_multimodal(mod1_tf_path=multiome_GEX_tf_path_test, mod2_tf_path=ATAC_tf_path_test,
                                mod1_network=GEX_network, mod2_network=ATAC_network, batch_size=batch_size2,
                                saved_weight_path=saved_weight_path_GEX, saved_weight_path2=saved_weight_path_ATAC, 
                                super_parameters=super_parameters, concat_modalities=concat_mod)
        
        if concat_mod:
            embedding12_train, _, GEX12_id_train = res12_train
            embedding12_test, _, GEX12_id_test = res12_test
            embedding13_train, _, GEX13_id_train = res13_train
            embedding13_test, _, GEX13_id_test = res13_test

            adata_merged_GEX = ad.concat([adata_cite_GEX[GEX12_id_train], adata_multiome_GEX[GEX13_id_train]], axis=0)
            embedding_merged = np.concatenate([embedding12_train, embedding13_train], axis=0)
            
            # Test QR on full appended GEX + ADT
            res_dict_12 = qr_test(adata_ref=adata_cite_GEX[GEX12_id_train], 
                    embedding_ref=embedding12_train, 
                    embedding_query=embedding12_test, 
                    test_cell_types=adata_cite_GEX_test[GEX12_id_test].obs['cell_type_l1'])
            res_dict_12["concat"] = "QR cite GEX + ADT"
            res_dict_12["epoch"] = epoch
            res_dict_12["pearsons"] = 0.0
            result_dicts.append(res_dict_12)
            
            res_dict_12 = qr_test(adata_ref=adata_merged_GEX, # Both train as reference
                    embedding_ref=embedding_merged, 
                    embedding_query=embedding12_test, 
                    test_cell_types=adata_cite_GEX_test[GEX12_id_test].obs['cell_type_l1'])
            res_dict_12["concat"] = "QR merged train, cite GEX + ADT"
            res_dict_12["epoch"] = epoch
            res_dict_12["pearsons"] = 0.0
            result_dicts.append(res_dict_12)

            # Test QR on full appended GEX + ATAC
            res_dict_13 = qr_test(adata_ref=adata_multiome_GEX[GEX13_id_train], 
                    embedding_ref=embedding13_train, 
                    embedding_query=embedding13_test, 
                    test_cell_types=adata_multiome_GEX_test[GEX13_id_test].obs['cell_type_l1'])
            res_dict_13["concat"] = "QR multiome GEX + ATAC"
            res_dict_13["epoch"] = epoch
            res_dict_13["pearsons"] = 0.0
            result_dicts.append(res_dict_13)

            res_dict_13 = qr_test(adata_ref=adata_merged_GEX, # both train as reference
                    embedding_ref=embedding_merged, 
                    embedding_query=embedding13_test, 
                    test_cell_types=adata_multiome_GEX_test[GEX13_id_test].obs['cell_type_l1'])
            res_dict_13["concat"] = "QR merged train, multiome GEX + ATAC"
            res_dict_13["epoch"] = epoch
            res_dict_13["pearsons"] = 0.0
            result_dicts.append(res_dict_13)

        else:
            embedding12_GEX_train, embedding12_ADT_train, _, GEX12_id_train = res12_train
            embedding12_GEX_test, embedding12_ADT_test, _, GEX12_id_test = res12_test
            embedding13_GEX_train, embedding13_ATAC_train, _, GEX13_id_train = res13_train
            embedding13_GEX_test, embedding13_ATAC_test, _, GEX13_id_test = res13_test

            adata_merged_GEX = ad.concat([adata_cite_GEX[GEX12_id_train], adata_multiome_GEX[GEX13_id_train]], axis=0)
            embedding_merged = np.concatenate([embedding12_GEX_train, embedding13_GEX_train], axis=0)
            print(f'After 1 {adata_merged_GEX.shape}')
            print(f'After 1 emb {embedding_merged.shape}')
            # Test QR on cite GEX
            res_dict_cite_GEX_12 = qr_test(adata_ref=adata_cite_GEX[GEX12_id_train], 
                    embedding_ref=embedding12_GEX_train, 
                    embedding_query=embedding12_GEX_test, 
                    test_cell_types=adata_cite_GEX_test[GEX12_id_test].obs['cell_type_l1'])
            res_dict_cite_GEX_12["concat"] = "QR cite GEX"
            res_dict_cite_GEX_12["epoch"] = epoch
            # Modality prediction ADT
            pearson = mp_3mod_test(embedding_train=embedding12_GEX_train, 
                                   embedding_test=embedding12_GEX_test, 
                                   adata_unknown_train=adata_ADT[GEX12_id_train], 
                                   adata_unknown_test=adata_ADT_test[GEX12_id_test])
            res_dict_cite_GEX_12["pearsons"] = pearson
            result_dicts.append(res_dict_cite_GEX_12)

            res_dict_cite_GEX_12 = qr_test(adata_ref=adata_merged_GEX, # Both train as reference
                    embedding_ref=embedding_merged, 
                    embedding_query=embedding12_GEX_test, 
                    test_cell_types=adata_cite_GEX_test[GEX12_id_test].obs['cell_type_l1'])
            res_dict_cite_GEX_12["concat"] = "QR merged cite GEX"
            res_dict_cite_GEX_12["epoch"] = epoch
            res_dict_cite_GEX_12["pearsons"] = 0.0
            result_dicts.append(res_dict_cite_GEX_12)

            # Test QR on multiome GEX
            res_dict_multiome_GEX_13 = qr_test(adata_ref=adata_multiome_GEX[GEX13_id_train], 
                    embedding_ref=embedding13_GEX_train, 
                    embedding_query=embedding13_GEX_test, 
                    test_cell_types=adata_multiome_GEX_test[GEX13_id_test].obs['cell_type_l1'])
            res_dict_multiome_GEX_13["concat"] = "QR multiome GEX"
            res_dict_multiome_GEX_13["epoch"] = epoch
            # Modality prediction ATAC
            pearson = mp_3mod_test(embedding_train=embedding13_GEX_train, 
                                   embedding_test=embedding13_GEX_test, 
                                   adata_unknown_train=adata_ATAC[GEX13_id_train], 
                                   adata_unknown_test=adata_ATAC_test[GEX13_id_test])
            res_dict_multiome_GEX_13["pearsons"] = pearson
            result_dicts.append(res_dict_multiome_GEX_13)

            res_dict_multiome_GEX_13 = qr_test(adata_ref=adata_merged_GEX, # Both train as reference
                    embedding_ref=embedding_merged, 
                    embedding_query=embedding13_GEX_test, 
                    test_cell_types=adata_multiome_GEX_test[GEX13_id_test].obs['cell_type_l1'])
            res_dict_multiome_GEX_13["concat"] = "QR merged multiome GEX"
            res_dict_multiome_GEX_13["epoch"] = epoch
            res_dict_multiome_GEX_13["pearsons"] = 0.0
            result_dicts.append(res_dict_multiome_GEX_13)

            # Test QR on multiome ATAC
            res_dict_multiome_qr_GEX_13 = qr_test(adata_ref=adata_ATAC[GEX13_id_train], 
                    embedding_ref=embedding13_ATAC_train, 
                    embedding_query=embedding13_ATAC_test, 
                    test_cell_types=adata_ATAC_test[GEX13_id_test].obs['cell_type_l1'])
            res_dict_multiome_qr_GEX_13["concat"] = "QR multiome ATAC"
            res_dict_multiome_qr_GEX_13["epoch"] = epoch
            res_dict_multiome_GEX_13["pearsons"] = 0.0
            result_dicts.append(res_dict_multiome_qr_GEX_13)

            # Modality prediction unknown ADT through GEX
            res_dict_cite_ADT_13 = mp_2mod_unknown_test(embedding_train=embedding12_GEX_train, 
                                                 embedding_test=embedding13_GEX_test, 
                                                 adata_unknown_train=adata_cite_GEX[GEX12_id_train], 
                                                 adata_unknown_test=adata_multiome_GEX_test[GEX13_id_test])
            res_dict_cite_ADT_13["concat"] = "MP ADT through GEX"
            res_dict_cite_ADT_13["epoch"] = epoch
            result_dicts.append(res_dict_cite_ADT_13)

            # Modality prediction unknown ATAC through GEX
            res_dict_cite_ATAC_13 = mp_2mod_unknown_test(embedding_train=embedding13_GEX_test, 
                                                 embedding_test=embedding12_GEX_train, 
                                                 adata_unknown_train=adata_multiome_GEX_test[GEX13_id_test], 
                                                 adata_unknown_test=adata_cite_GEX[GEX12_id_train])
            res_dict_cite_ATAC_13["concat"] = "MP ATAC through GEX"
            res_dict_cite_ATAC_13["epoch"] = epoch
            result_dicts.append(res_dict_cite_ATAC_13)

            print(f'After 1 {adata_merged_GEX.shape}')
            print(f'After 1 emb {embedding_merged.shape}')

            # Merged modality prediction unknown ADT through GEX
            res_dict_cite_ADT_13 = mp_2mod_unknown_test(embedding_train=embedding_merged, 
                                                 embedding_test=embedding13_GEX_test, 
                                                 adata_unknown_train=adata_merged_GEX, 
                                                 adata_unknown_test=adata_multiome_GEX_test[GEX13_id_test])
            res_dict_cite_ADT_13["concat"] = "MP merged ADT through GEX"
            res_dict_cite_ADT_13["epoch"] = epoch
            result_dicts.append(res_dict_cite_ADT_13)

            # Merged modality prediction unknown ATAC through GEX
            res_dict_cite_ATAC_13 = mp_2mod_unknown_test(embedding_train=embedding_merged, 
                                                 embedding_test=embedding12_GEX_train, 
                                                 adata_unknown_train=adata_merged_GEX, 
                                                 adata_unknown_test=adata_cite_GEX[GEX12_id_train])
            res_dict_cite_ATAC_13["concat"] = "MP merged ATAC through GEX"
            res_dict_cite_ATAC_13["epoch"] = epoch
            result_dicts.append(res_dict_cite_ATAC_13)

            # TODO ATAC -> GEX -> ADT
            
            return result_dicts



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
    batch_size = args.batch_size
    batch_size2 = args.batch_size2
    drop_rate = args.drop_rate
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
        GEX_cite_tf_path, ADT_cite_tf_path, adata_GEX_cite, adata_ADT_cite, GEX_cite_tf_path_test, ADT_cite_tf_path_test, adata_GEX_cite_test, adata_ADT_cite_test, GEX_multiome_tf_path, ATAC_multiome_tf_path, adata_GEX_multiome, adata_ATAC_multiome, GEX_multiome_tf_path_test, ATAC_multiome_tf_path_test, adata_GEX_multiome_test, adata_ATAC_multiome_test = read_data(data=data, save_path=save_path, task=task)

        # Train
        weight_path = save_path + 'weight/'
        if train:
            train_cellbind(data=data, weight_path=weight_path, 
                           GEX_cite_tf_path=GEX_cite_tf_path, ADT_cite_tf_path=ADT_cite_tf_path, 
                           GEX_multiome_tf_path=GEX_multiome_tf_path, ATAC_multiome_tf_path=ATAC_multiome_tf_path,
                           attention_t=attention_t, attention_s=attention_s, 
                           batch_size=batch_size, batch_size2=batch_size2, epoch=epoch, lr=lr, drop_rate=drop_rate, 
                           heads=heads, combine_omics=combine_omics, model_type=model_type)
        print("Trained.")

        if test:   
            # epochs_test = [epoch]

            epochs_test = []
            i = 4
            while i < epoch:
                epochs_test.append(i)
                i = i * 2
            epochs_test.append(epoch)

            res_dicts = None
            for e in epochs_test:
                res_dicts_tmp = test_cellbind(adata_cite_GEX=adata_GEX_cite, adata_cite_GEX_test=adata_GEX_cite_test,
                                        adata_ADT=adata_ADT_cite, adata_ADT_test=adata_ADT_cite_test,
                                        adata_multiome_GEX=adata_GEX_multiome, adata_multiome_GEX_test=adata_GEX_multiome_test, 
                                        adata_ATAC=adata_ATAC_multiome, adata_ATAC_test=adata_ATAC_multiome_test,
                                        cite_GEX_tf_path=GEX_cite_tf_path, multiome_GEX_tf_path=GEX_multiome_tf_path, 
                                        ADT_tf_path=ADT_cite_tf_path, ATAC_tf_path=ATAC_multiome_tf_path, 
                                        cite_GEX_tf_path_test=GEX_cite_tf_path_test, multiome_GEX_tf_path_test=GEX_multiome_tf_path_test, 
                                        ADT_tf_path_test=ADT_cite_tf_path_test, ATAC_tf_path_test=ATAC_multiome_tf_path_test, 
                                        weight_path=weight_path, data=data,
                                        attention_t=attention_t, attention_s=attention_s, 
                                        batch_size=batch_size, batch_size2=batch_size2, 
                                        epoch=e, lr=lr, drop_rate=drop_rate, 
                                        heads=heads, combine_omics=combine_omics, model_type=model_type)
                if res_dicts is None:
                    res_dicts = res_dicts_tmp
                else:
                    res_dicts.extend(res_dicts_tmp)
                
            res_df = pd.DataFrame(res_dicts)
            res_df.to_csv(f'./Multimodal_pretraining/results/{data}_{combine_omics}_mt_{model_type}_bs_{batch_size}_{batch_size2}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}_{repeat}.csv')

main()
