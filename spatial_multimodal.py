import argparse
from collections import Counter
import math
import os
import sys

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
sys.path.append("../")
# from concerto_function5_3 import *
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
from scipy.sparse import issparse
import scipy

def preprocessing_changed_rna(
        adata,
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = 10000,
        n_top_features=2000,  # or gene list
        chunk_size: int = 20000,
        is_hvg = True,
        batch_key = 'batch',
        log=True
):
    if min_features is None: min_features = 600
    if n_top_features is None: n_top_features = 40000

    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    adata = adata[:, [gene for gene in adata.var_names
                      if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]

    sc.pp.filter_cells(adata, min_genes=min_features)

    sc.pp.filter_genes(adata, min_cells=min_cells)

    sc.pp.normalize_total(adata, target_sum=target_sum)

    sc.pp.log1p(adata)
    
    if is_hvg == True:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key=batch_key, inplace=True, subset=True)

    print('Processed dataset shape: {}'.format(adata.shape))
    return adata

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

def concerto_make_tfrecord(processed_ref_adata, tf_path, batch_col_name=None):
    # 有输入batch_col_name的时候，用这列作为batchid， 若无假设所有是一个batch
	# 不做乱序,
    if batch_col_name is None:
        batch_col_name = 'batch_'
        sample_num = len(processed_ref_adata.obs_names.tolist())
        processed_ref_adata.obs[batch_col_name]  = ['0']*sample_num
    print(processed_ref_adata)
    batch_list = processed_ref_adata.obs[batch_col_name].unique().tolist()
    cc = dict(Counter(batch_list))
    cc = list(cc.keys())
    tfrecord_file = tf_path + '/tf.tfrecord'
    if not os.path.exists(tf_path):
        os.makedirs(tf_path)
    create_tfrecord(processed_ref_adata, cc, tfrecord_file, zero_filter=False, norm=True, batch_key =batch_col_name)

    return tf_path

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def _bytes_feature_image(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example_batch(x_feature, x_weight, y_batch, x_id, cell_id):

    feature = {
        'feature': _int64_feature(x_feature),
        'value': _float_feature(x_weight),
        'batch': _int64_feature(y_batch),
        'id': _bytes_feature(x_id),
        'cell_id': _bytes_feature(cell_id)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_example_batch_spatial(x_feature, x_id):

    feature = {
        'image_raw': _bytes_feature_image(x_feature),
        'id': _bytes_feature(x_id),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def create_tfrecord(source_file,  batch_dict, tfrecord_file, zero_filter=False, norm=False, batch_key = 'batch'):
    if type(source_file.X) != np.ndarray:
        x_data = source_file.X.toarray()
    else:
        x_data = source_file.X
    batch_data = source_file.obs[batch_key].tolist()
    obs_name_list = source_file.obs_names.tolist()
    cell_ids = source_file.obs["cell_id"].tolist()
    batch_number = []
    for j in range(len(batch_data)):
        batch = batch_data[j]
        place = batch_dict.index(batch)
        batch_number.append(place)

    counter = 0
    batch_examples = {}
    for x, batch, k, cell_id in zip(x_data, batch_number, obs_name_list, cell_ids):
        if zero_filter is False:
            x = x + 10e-6
            indexes = np.where(x >= 0.0)
        else:
            indexes = np.where(x > 0.0)
        values = x[indexes]

        features = np.array(indexes)
        features = np.reshape(features, (features.shape[1]))
        values = np.array(values, dtype=float)
        # values = values / np.linalg.norm(values)

        if batch not in batch_examples:
            batch_examples[batch] = []

        example = serialize_example_batch(features, values, np.array([int(batch)]), k, cell_id)
        batch_examples[batch].append(example)

        counter += 1
        if counter % 1000 == 0:
            print('counter: {} shape: {}, batch: {}'.format(counter, features.shape, batch))

            #print(x)
            #print(values)
            #print("batchs: ", batch_dict)

    for item in batch_examples.items():
        batch = item[0]
        examples = item[1]
        if zero_filter is False:
            file = tfrecord_file.replace('.tfrecord', '_{}.tfrecord'.format(batch))
        else:
            if norm is False:
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero_no_norm.tfrecord'.format(batch))
            else:
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero.tfrecord'.format(batch))
        with tf.io.TFRecordWriter(file) as writer:
            for example in examples:
                writer.write(example)
    save_dict = {'vocab size': len(features)}
    file = tfrecord_file.replace('tf.tfrecord','vocab_size.npz')
    np.savez_compressed(file, **save_dict)
#     np.savez_compressed('vocab_size.npz', **save_dict)

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))

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

    sc.pp.subsample(sdata['table'], random_state=42, n_obs=20000)
    adata_RNA = sdata['table']
    # Create PCA for benchmarking
    sc.tl.pca(adata_RNA)

    adata_RNA.obs["batch"] = np.full((adata_RNA.shape[0],), 1)

    adata_RNA = preprocessing_changed_rna(adata_RNA, min_features = 0, is_hvg=is_hvg_RNA, batch_key='batch')
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

    tfrecord_file = RNA_tf_path + '/tf_0.tfrecord'
    if not os.path.exists(RNA_tf_path):
        os.makedirs(RNA_tf_path)
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for geom in adata_RNA.obs['cell_id']:
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
            example = serialize_example_batch_spatial(image_raw, geom)
            writer.write(example)

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

    print("Read data")

    RNA_tf_path, adata_RNA, staining_tf_path  = prepare_data_spatial(sdata=sdata, save_path=save_path, is_hvg_RNA=False)

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

    # # Check num GPUs
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(f"\nAvailable GPUs: {gpus}\n")
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    
    # Read data
    save_path = './Multimodal_pretraining/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    RNA_tf_path, adata_RNA, staining_tf_path = read_data_spatial(data=data, save_path=save_path)

main()
