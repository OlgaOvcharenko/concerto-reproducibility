import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.model_selection import (
    train_test_split,
)
import matplotlib.pyplot as plt
# Inital setting for plot size
from matplotlib import rcParams
import h5py

# # path_tmp = "../../../../../Downloads/GSE158013_RAW/GSM5123955_X066-RP0C1W1_leukopak_perm-cells_cite_200M_rna_counts.h5"
# path_tmp = "../../../../../Downloads/GSE158013_RAW/GSM4949911_X061-AP0C1W1_leukopak_perm-cells_tea_fulldepth_rna_counts.h5"
# f = h5py.File(path_tmp, 'r')
# print(list(f.keys()))
# print(f['matrix'])
# print(list(f['matrix'].keys()))
# print(f['well'])
# print(list(f['well'].keys()))


# adata_rna = sc.read_10x_h5(path_tmp)
# print(adata_rna)
# print(adata_rna.var["feature_types"].unique())


# import pandas as pd
# data = pd.read_csv('../../../../../Downloads/GSE158013_RAW/GSM4949911_X061-AP0C1W1_leukopak_perm-cells_tea_fulldepth_adt_counts.csv.gz', nrows=100, compression='gzip',)
# print(list(data.columns))


# data = pd.read_csv('../../../../../Downloads/GSE158013_RAW/GSM4949911_X061-AP0C1W1_leukopak_perm-cells_tea_fulldepth_cellranger-arc_per_barcode_metrics.csv.gz', nrows=100, compression='gzip',)
# print(data)
# print(list(data.columns))

path = '../../../../../Downloads/GSE158013_RAW/GSM5123952_X066-MP0C1W4_leukopak_perm-cells_tea_200M_atac_filtered_metadata.csv.gz'
data = pd.read_csv(path, nrows=100, compression='gzip',)
print(data)
print(list(data.columns))


