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

# X_train = np.random.rand(1000, 128)
# X_test = np.random.rand(250, 128)
# y_train = np.random.randint(low=1, high=2, size=1000)
# y_test = np.random.randint(low=1, high=3, size=250)

# adata_new = ad.AnnData(np.append(X_train, X_test, axis=0))
# print(adata_new)
# sc.pp.neighbors(adata_new, metric="cosine", use_rep='X')
# sc.tl.leiden(adata_new, resolution=0.2)

# print(adata_new)


# # from scib_metrics.benchmark import Benchmarker

# adata_gex = sc.read_h5ad("./Multimodal_pretraining/data/data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")
# print(adata_gex)

# print(adata_gex.X)
# print(adata_gex.var["feature_types"])
# print(adata_gex.var["feature_types"].value_counts())
# print(adata_gex.layers["counts"])
# print(adata_gex.obs["batch"])

adata_adt = sc.read_h5ad("./Multimodal_pretraining/data/data/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad")
# print(adata_adt.var["feature_types"])
# print(adata_adt.var["feature_types"].value_counts())
# adata_adt.X = adata_adt.layers["counts"]
# adata_adt_atac = adata_adt[:, 13431:]
# adata_adt_gex = adata_adt[:, 0:13431]
print(adata_adt)
# print(np.unique(adata_adt.obs["batch"].to_list()))
# print(np.unique(adata_adt.obs["cell_type"].to_list()))

# print(adata_adt_atac)
# print(adata_adt_gex)

# # print(adata_adt.X)
# # print(adata_adt.X)

# # print(np.unique(adata_adt_rna.obs["batch"].to_list()))
# # print(np.unique(adata_adt_rna.obs["cell_type"].to_list()))


# # b_list = np.unique(adata_adt_rna.obs["batch"].to_list())
# # res = np.unique(adata_adt_rna[adata_adt_rna.obs["batch"] == b_list[0], :].obs["cell_type"].to_list())
# # for b in b_list[1:]:
# #     print(b)
# #     tmp_res = np.unique(adata_adt_rna[adata_adt_rna.obs["batch"] == b, :].obs["cell_type"].to_list())
# #     if len(res) != len(tmp_res):
# #         print(False)
# #     else:
# #         print(all(res == tmp_res))
# #     print(np.unique(adata_adt_rna[adata_adt_rna.obs["batch"] == b, :].obs["cell_type"].to_list()))


# # # print(adata_adt.obs["cell_type"])

# # # print("-----"*10)

# # # print(adata_adt.obs["cell_type"].unique())

# # # print("-----"*10)

# # # print(adata_adt.X)



# # # print("-----"*10)

# # # print(adata_adt_cite)
# # # print(adata_adt_cite.X)

# # # print("-----"*10)

# # # print(adata_adt_rna)
# # # print(adata_adt_rna.X)


# path = './Multimodal_pretraining/data/data/multi_gene_l2.loom'
# adata_RNA = sc.read(path)

# print(adata_RNA)

# path = './Multimodal_pretraining/data/data/multi_protein_l2.loom'
# adata_Protein = sc.read(path) #cell_type batch

# print(adata_Protein)

# # train_idx, test_idx = train_test_split(
# #     adata_RNA.obs_names.values,
# #     test_size=0.3,
# #     stratify=adata_RNA.obs["batch"],
# #     shuffle=True,
# #     random_state=42,
# # )

# # adata_RNA_train = adata_RNA[train_idx, :]
# # adata_RNA_test = adata_RNA[test_idx, :]

# # print(adata_RNA)

# # print(np.unique(adata_RNA.obs["batch"].to_list()))
# # print(np.unique(adata_RNA.obs["cell_type"].to_list()))
# # print(np.unique(adata_RNA.obs["batch"].to_list()))
# # print(np.unique(adata_RNA.obs["cell_type"].to_list()))

# # train_idx = (adata_RNA.obs["batch"] != "P6") & (adata_RNA.obs["batch"] != "P7") & (adata_RNA.obs["batch"] != "P8")
# # print(np.unique(adata_RNA[train_idx, :].obs["batch"]))
# # print(np.unique(adata_RNA[(train_idx != 1), :].obs["batch"]))

# # b_list = np.unique(adata_RNA.obs["batch"].to_list())
# # res = np.unique(adata_RNA[adata_RNA.obs["batch"] == b_list[0], :].obs["cell_type"].to_list())
# # for b in b_list[1:]:
# #     print(b)
# #     print((res == np.unique(adata_RNA[adata_RNA.obs["batch"] == b, :].obs["cell_type"].to_list())))
# #     print(np.unique(adata_RNA[adata_RNA.obs["batch"] == b, :].obs["cell_type"].to_list()))


# # adata_Protein = adata_Protein[train_idx, :]

# # adata_RNA_test = adata_RNA[test_idx, :]
# # adata_Protein_test = adata_Protein[test_idx, :]

# # Create PCA for benchmarking
# # adata_merged_tmp = ad.concat([adata_RNA, adata_Protein], axis=1)
# # # adata_merged.var_names_make_unique()
# # adata_merged.obs = adata_RNA.obs
# # adata_merged.obsm = adata_RNA.obsm
# # path = './Multimodal_pretraining/data/simulated_train_0_mt_3_bs_64_100_0.001_0.1_False_True_128.h5ad'
# # adata_RNA = sc.read(path)
# # print(adata_RNA)
                


# # # # print("Read simulated data")
# # # # print(adata_Protein)
# # # # print("Read obsm")
# # # # print(list(adata_Protein.obsm.keys()))

# # # # bm = Benchmarker(
# # # #     adata_merged,
# # # #     batch_key="batch",
# # # #     label_key="cell_type",
# # # #     embedding_obsm_keys=["Unintegrated"],
# # # #     n_jobs=10,
# # # # )
# # # # bm.benchmark()
# # # # bm.plot_results_table(save_dir=f'./Multimodal_pretraining/plots/test.png')

# # # # df = bm.get_results(min_max_scale=False)
# # # # print(df)


# from matplotlib import gridspec, pyplot as plt
# from sklearn.preprocessing import OrdinalEncoder


# def make_plot():
#     path = 'Multimodal_pretraining/data/simulated_train_0_mt_3_bs_64_100_0.001_0.1_False_True_128.h5ad'
#     adata_RNA = sc.read(path)
#     print(adata_RNA)
#     l2tol1 = {
#                     'CD8 Naive': 'CD8 T',
#                     'CD8 Proliferating': 'CD8 T',
#                     'CD8 TCM': 'CD8 T',
#                     'CD8 TEM': 'CD8 T',
#                     'CD4 CTL': 'CD4 T',
#                     'CD4 Naive': 'CD4 T',
#                     'CD4 Proliferating': 'CD4 T',
#                     'CD4 TCM': 'CD4 T',
#                     'CD4 TEM': 'CD4 T',
#                     'Treg': 'CD4 T',
#                     'NK': 'NK',
#                     'NK Proliferating': 'NK',
#                     'NK_CD56bright': 'NK',
#                     'dnT': 'other T',
#                     'gdT': 'other T',
#                     'ILC': 'other T',
#                     'MAIT': 'other T',
#                     'CD14 Mono': 'Monocytes',
#                     'CD16 Mono': 'Monocytes',
#                     'cDC1': 'DC',
#                     'cDC2': 'DC',
#                     'pDC': 'DC',
#                     'ASDC':'DC',
#                     'B intermediate': 'B',
#                     'B memory': 'B',
#                     'B naive': 'B',
#                     'Plasmablast': 'B',
#                     'Eryth': 'other',
#                     'HSPC': 'other',
#                     'Platelet': 'other'
#                 }

#     adata_RNA.obs['cell_type_l1'] = adata_RNA.obs['cell_type'].map(l2tol1)

#     dpi = 1000
#     fig_size = ((7.125 - 0.17), ((7.125 - 0.17) / 1.8) / 1.618)

#     fig = plt.figure(
#         constrained_layout=True,
#         figsize=fig_size,
#         dpi=dpi,
#         facecolor="w",
#         edgecolor="k",
#     )
#     spec2 = gridspec.GridSpec(
#         ncols=4,
#         nrows=1,
#         figure=fig,
#         left=0.02,
#         right=1,
#         top=0.77,
#         bottom=0.06,
#         wspace=0.15,
#     )
#     ax1 = fig.add_subplot(spec2[0])
#     ax2 = fig.add_subplot(spec2[1])
#     ax3 = fig.add_subplot(spec2[2])

#     axi = [ax1, ax2, ax3]

#     # ord_enc = OrdinalEncoder()
#     # r = ord_enc.fit_transform(pd.DataFrame(adata_RNA.obs['batch'].to_list()))
    
#     # print(ord_enc.categories_)

#     sci = axi[0].scatter(
#         adata_RNA.obsm["X_pca"][:, 0],
#         adata_RNA.obsm["X_pca"][:, 1],
#         marker=".",
#         c=adata_RNA.obs["batch"],
#         cmap="Accent",
#         zorder=0,
#         alpha=0.2,
#     )

#     axi[0].set_yticks([])
#     axi[0].set_xticks([])
#     axi[0].set_ylabel("UMAP2", size=8)
#     axi[0].set_xlabel("UMAP1", size=8)
#     axi[0].set_title("Global clock", size=8)
#     axi[0].yaxis.set_label_coords(x=-0.01, y=0.5)
#     axi[0].xaxis.set_label_coords(x=0.5, y=-0.02)

#     axi[2].set_yticks([])
#     axi[2].set_xticks([])
#     axi[2].set_ylabel("UMAP2", size=8)
#     axi[2].set_xlabel("UMAP1", size=8)
#     axi[2].set_title("Inter-group clock", size=8)
#     axi[2].yaxis.set_label_coords(x=-0.01, y=0.5)
#     axi[2].xaxis.set_label_coords(x=0.5, y=-0.02)

#     plt.savefig("test.jpg")

# # make_plot()
