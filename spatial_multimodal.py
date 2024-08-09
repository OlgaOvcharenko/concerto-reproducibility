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
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import KernelPCA
import psutil
import spatialdata
import spatialdata_io
from spatialdata_io.readers.xenium import xenium_aligned_image
import numpy as np
import matplotlib.pyplot as plt
from spatialdata import rasterize
from PIL import Image

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

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def fix_image_size(width, height, x_min, x_max, y_min, y_max):
    slice_width_to_add = width - (x_max - x_min)
    slice_height_to_add = height - (y_max - y_min)

    if slice_width_to_add % 2 == 0:
        x_min -= slice_width_to_add / 2
        x_max += slice_width_to_add / 2

    else:
        x_min -= math.floor(slice_width_to_add / 2)
        x_max += slice_width_to_add - math.floor(slice_width_to_add / 2)

    if slice_height_to_add % 2 == 0:
        y_min -= slice_height_to_add / 2
        y_max += slice_height_to_add / 2

    else:
        y_min -= math.floor(slice_height_to_add / 2)
        y_max += slice_height_to_add - math.floor(slice_height_to_add / 2)


    if int(x_max - x_min) != 128:
        print(x_min, x_max)
        x_max += int(x_max - x_min)
        

    if int(y_max - y_min) != 128:
        print(y_min, y_max)
        y_max += int(y_max - y_min)

    return x_min, x_max, y_min, y_max

def prepare_data_spatial(sdata, save_path: str = '', is_hvg_RNA: bool = False):
    print("Read spatial data.")

    # Create PCA for benchmarking
    sc.tl.pca(adata_RNA)

    sdata["table"].obs["batch"] = np.full(sdata["table"].shape, 1)

    adata_RNA = preprocessing_changed_rna(sdata["table"], min_features = 0, is_hvg=is_hvg_RNA, batch_key='batch')
    print(f"RNA data shape {adata_RNA.shape}")
    
    adata_RNA.write_h5ad(save_path + f'spatial_adata_RNA.h5ad')
    print("Saved adata.")

    path_file = 'tfrecord/'
    RNA_tf_path = save_path + path_file + 'spatial_RNA_tf/'
    RNA_tf_path = concerto_make_tfrecord(adata_RNA, tf_path=RNA_tf_path, batch_col_name='batch')
    print("Made tf record RNA.")

    rows = 128
    cols = 128
    depth = 3

    staining_tf_path = save_path + path_file + 'spatial_staining_tf/'
    print('Writing ', staining_tf_path)
    writer = tf.python_io.TFRecordWriter(staining_tf_path)
    for geom in sdata["cell_boundaries"].index:
        coords_x, coords_y = spatialdata.transform(sdata["cell_boundaries"], to_coordinate_system="global").loc[geom, "geometry"].exterior.coords.xy
        x_min, y_min = np.min(coords_x), np.min(coords_y)
        x_max, y_max = np.max(coords_x), np.max(coords_y)

        x_min, x_max, y_min, y_max = fix_image_size(rows, cols, x_min, x_max, y_min, y_max)

        res = rasterize(
            sdata["he_image"],
            ["x", "y"],
            min_coordinate=[x_min, y_min],
            max_coordinate=[x_max, y_max],
            target_unit_to_pixels=1.0,
            target_coordinate_system="global"
        )

        image_raw = res.to_numpy().transpose(1,2,0).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _bytes_feature(geom),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())

    return RNA_tf_path, adata_RNA, staining_tf_path

def read_data_spatial(data: str = "", save_path: str = ""):
    if data != 'spatial':
        raise Exception('[SPATIAL] Incorrect dataset name.')

    data_path="./Multimodal_pretraining/data/Xinium/Xenium_V1_humanLung_Cancer_FFPE_outs"
    alignment_matrix_path = "./Multimodal_pretraining/data/Xinium/Xenium_V1_humanLung_Cancer_FFPE_he_imagealignment.csv"
    he_path = "./Multimodal_pretraining/data/Xinium/Xenium_V1_humanLung_Cancer_FFPE_he_image.ome.tif"

    sdata = spatialdata_io.xenium(data_path)

    image = xenium_aligned_image(he_path, alignment_matrix_path)
    sdata['he_image'] = image

    RNA_tf_path, adata_RNA, staining_tf_path  = prepare_data_spatial(sdata=sdata, save_path=save_path, is_hvg_RNA=False)

    return RNA_tf_path, adata_RNA, staining_tf_path

def train_concerto(weight_path: str, RNA_tf_path: str, Protein_tf_path: str, data: str, 
                   attention_t: bool, attention_s: bool,
                   batch_size:int, epoch: int, lr: float, drop_rate: float, 
                   heads: int, combine_omics: int, model_type: int):
    # Train
    if attention_t == True and attention_s == False:
        concerto_train_multimodal(['RNA','Protein'] if data == 'simulated' else ['GEX', 'ATAC'], 
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
        concerto_train_multimodal_tt(['RNA','Protein'] if data == 'simulated' else ['GEX', 'ATAC'],
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
        concerto_train_multimodal_ss(['RNA','Protein'] if data == 'simulated' else ['GEX', 'ATAC'],
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
    diverse_tests_names = []

    only_RNAs = [True, False] if combine_omics == 1 else [False]
    for only_RNA in only_RNAs:
        for dr in [drop_rate]:
            for nn in ["encoder"]:
                for e in ep_vals: 
                    saved_weight_path = f'./Multimodal_pretraining/weight/multi_weight_{nn}_{data}_{batch_size}_model_{combine_omics}_{model_type}_epoch_{e}_{lr}_{drop_rate}_{attention_t}_{attention_s}_{heads}.h5'

                    if (nn == "decoder" and attention_s == False) or (nn == "encoder" and attention_t == False):
                        embedding, batch, RNA_id, attention_weight =  concerto_test_multimodal_decoder(
                        ['RNA','Protein'] if data == 'simulated' else ['GEX', 'ATAC'],
                        weight_path, 
                        RNA_tf_path_test,
                        Protein_tf_path_test,
                        n_cells_for_sample=None,
                        super_parameters={
                            'batch_size': batch_size, 
                            'epoch': e, 'lr': lr, 
                            'drop_rate': dr, 
                            'attention_t': attention_t, 
                            'attention_s': attention_s, 
                            'heads': heads,
                            'combine_omics': combine_omics
                        }, 
                        saved_weight_path = saved_weight_path)
                    else:
                        embedding, batch, RNA_id, attention_weight =  concerto_test_multimodal(
                            ['RNA','Protein'] if data == 'simulated' else ['GEX', 'ATAC'],
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
                                'model_type': model_type
                            }, 
                            saved_weight_path = saved_weight_path,
                            only_RNA=only_RNA)

                    print("Tested.")
                    
                    if data == "simulated":
                        adata_RNA = sc.read(save_path + f'adata_RNA_{"train" if train else "test"}.h5ad')
                    else:
                        adata_RNA = sc.read(save_path + f'adata_gex_{"train" if train else "test"}.h5ad')
                    
                    adata_RNA_1 = adata_RNA[RNA_id]
                    adata_RNA_1.obsm['X_embedding'] = embedding

                    print(f"Shape of the embedding {embedding.shape}.")
                    
                    adata_merged = adata_merged[RNA_id]

                    adata_merged.obsm[f'{"train" if train else "test"}_{e}_{nn}_{dr}_{only_RNA}'] = embedding
                    diverse_tests_names.append(f"{train}_{e}_{nn}_{dr}_{only_RNA}")

                    l2tol1 = {
                        'CD8 Naive': 'CD8 T',
                        'CD8 Proliferating': 'CD8 T',
                        'CD8 TCM': 'CD8 T',
                        'CD8 TEM': 'CD8 T',
                        'CD8+ T': 'CD8 T',
                        'CD8+ T naive': 'CD8 T',
                        'CD4 CTL': 'CD4 T',
                        'CD4 Naive': 'CD4 T',
                        'CD4 Proliferating': 'CD4 T',
                        'CD4 TCM': 'CD4 T',
                        'CD4 TEM': 'CD4 T',
                        'CD4+ T activated': 'CD4 T',
                        'CD4+ T naive': 'CD4 T',
                        'CD14+ Mono': 'CD14 T',
                        'CD16+ Mono': 'CD16 T',
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
                        'B1 B': 'B',
                        'Plasmablast': 'B',
                        'Eryth': 'other',
                        'HSPC': 'other',
                        'Platelet': 'other'
                    }

                    # if data == 'simulated':
                    adata_RNA_1.obs['cell_type_l1'] = adata_RNA_1.obs['cell_type'].map(l2tol1)
                    adata_merged.obs['cell_type_l1'] = adata_RNA_1.obs['cell_type'].map(l2tol1)
                    # else:
                    #     adata_RNA_1.obs['cell_type_l1'] = adata_RNA_1.obs['cell_type']
                    #     adata_merged.obs['cell_type_l1'] = adata_RNA_1.obs['cell_type']
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

                    if not train:
                        print("Predict")
                        # adata_RNA_1.obs[f'pred_cell_type_{e}_{nn}_{dr}_{only_RNA}'] = query_to_reference(X_train=adata_merged_train.obsm[f'train_{e}_{nn}_{dr}'], y_train=adata_merged_train.obs["cell_type_l1"], X_test=adata_merged.obsm[f'test_{e}_{nn}_{dr}'], y_test=adata_merged.obs["cell_type_l1"], ).set_index(adata_RNA_1.obs_names)["val_ct"]
                        query_neighbor, _ = knn_classifier(ref_embedding=adata_merged_train.obsm[f'train_{e}_{nn}_{dr}_{only_RNA}'], query_embedding=embedding, ref_anndata=adata_merged_train, column_name='cell_type_l1', k=5)

                        acc = accuracy_score(adata_merged.obs['cell_type_l1'].to_list(), query_neighbor)
                        f1 = f1_score(adata_merged.obs['cell_type_l1'].to_list(), query_neighbor, average=None)
                        f1_median = np.median(f1)
                        print('acc:{:.2f} f1-score:{:.2f}'.format(acc,f1_median))

                        adata_RNA_1.obs[f'pred_cell_type_{e}_{nn}_{dr}_{only_RNA}'] = query_neighbor
                        adata_merged.obs[f'pred_cell_type_{e}_{nn}_{dr}_{only_RNA}'] = query_neighbor

                    # sc.pp.neighbors(adata_RNA_1, use_rep='X_embedding', metric='cosine')
                    sc.tl.leiden(adata_RNA_1, resolution=0.2)
                    sc.tl.umap(adata_RNA_1, min_dist=0.1)
                    adata_merged.obsm[f'{"train" if train else "test"}_umap_{e}_{nn}_{dr}'] = adata_RNA_1.obsm["X_umap"]
                    adata_merged.obs[f'{"train" if train else "test"}_leiden_{e}_{nn}_{dr}'] = adata_RNA_1.obs["leiden"]
                    sc.set_figure_params(dpi=150)

                    if not train:
                        color=['cell_type_l1', f'pred_cell_type_{e}_{nn}_{dr}_{only_RNA}', 'leiden', 'batch']
                        # color=['cell_type_l1', 'leiden', 'batch']
                    else:
                        color=['cell_type_l1', 'leiden', 'batch']

                    sc.pl.umap(adata_RNA_1, color=color, legend_fontsize ='xx-small', size=5, legend_fontweight='light', edges=True)
                    plt.savefig(f'./Multimodal_pretraining/plots/{data}/{data}_knn_concerto_{"train" if train else "test"}_{combine_omics}_oRNA{only_RNA}_mt_{model_type}_bs_{batch_size}_{nn}_{e}_{lr}_{drop_rate}_{dr}_{attention_s}_{attention_t}_{heads}.png')
                    
                    # scv.pl.velocity_embedding(f'./Multimodal_pretraining/plots/{data}/{data}_mt_{model_type}_bs_{batch_size}_{nn}_{e}_{lr}_{drop_rate}_{dr}_{attention_s}_{attention_t}_{heads}.png', basis="umap")

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

    print(f"Multimodal correction: epoch {epoch}, model type {model_type}, lr {lr}, batch_size {batch_size}, drop_rate {drop_rate}, attention_t {attention_t}, attention_s {attention_s}, heads {heads}.")

    # Check num GPUs
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(f"\nAvailable GPUs: {gpus}\n")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    
    # Read data
    save_path = './Multimodal_pretraining/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    RNA_tf_path, adata_RNA, staining_tf_path = read_data_spatial(data=data, save_path=save_path)

    # Train
    # weight_path = save_path + 'weight/'
    # if train:
    #     train_concerto(weight_path=weight_path, RNA_tf_path=RNA_tf_path, Protein_tf_path=Protein_tf_path, data=data, 
    #                attention_t=attention_t, attention_s=attention_s, 
    #                batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
    #                heads=heads, combine_omics=combine_omics, model_type=model_type)
    # print("Trained.")

    # if test:
    #     # Test on train data
    #     adata_merged = test_concerto(weight_path=weight_path, RNA_tf_path_test=RNA_tf_path, Protein_tf_path_test=Protein_tf_path, data=data, 
    #                attention_t=attention_t, attention_s=attention_s,
    #                batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
    #                heads=heads, combine_omics=combine_omics, model_type=model_type, 
    #                save_path=save_path, train=True, adata_merged=adata_merged, adata_RNA=adata_RNA)
        
    #     filename = f'./Multimodal_pretraining/data/{data}/{data}_train_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'
    #     save_merged_adata(adata_merged=adata_merged, filename=filename)

    #     # Test on test data
    #     adata_merged_test = test_concerto(weight_path=weight_path, RNA_tf_path_test=RNA_tf_path_test, Protein_tf_path_test=Protein_tf_path_test, data=data, 
    #                attention_t=attention_t, attention_s=attention_s,
    #                batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
    #                heads=heads, combine_omics=combine_omics, model_type=model_type, 
    #                save_path=save_path, train=False, adata_merged=adata_merged_test, adata_RNA=adata_RNA_test, adata_merged_train=adata_merged)

    #     filename = f'./Multimodal_pretraining/data/{data}/{data}_test_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'
    #     save_merged_adata(adata_merged=adata_merged_test, filename=filename)

main()
