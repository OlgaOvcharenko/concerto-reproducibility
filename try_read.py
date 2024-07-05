import scanpy as sc
import anndata as ad
from sklearn.model_selection import (
    train_test_split,
)
# from scib_metrics.benchmark import Benchmarker

# adata_gex = sc.read_h5ad("./Multimodal_pretraining/data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")
# adata_adt = sc.read_h5ad("./Multimodal_pretraining/data/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad")
# adata_adt_cite = adata_adt[:, 0:13431]
# adata_adt_rna = adata_adt[:, 13431:]

# print(adata_adt.shape)
# print(adata_adt)

# print(adata_adt.obs["cell_type"])

# print("-----"*10)

# print(adata_adt.obs["cell_type"].unique())

# print("-----"*10)

# print(adata_adt.X)



# print("-----"*10)

# print(adata_adt_cite)
# print(adata_adt_cite.X)

# print("-----"*10)

# print(adata_adt_rna)
# print(adata_adt_rna.X)


path = './Multimodal_pretraining/data/multi_gene_l2.loom'
adata_RNA = sc.read(path)

train_idx, test_idx = train_test_split(
    adata_RNA.obs_names.values,
    test_size=0.3,
    stratify=adata_RNA.obs["cell_type"],
    shuffle=True,
    random_state=42,
)

print(
    f"Train set : \n {adata_RNA[train_idx, :].obs.cell_type.value_counts()} \n \n Test set: \n {adata_RNA[test_idx, :].obs.cell_type.value_counts()}"
)

path = './Multimodal_pretraining/data/multi_protein_l2.loom'
adata_Protein = sc.read(path) #cell_type batch

print(
    f"Train set : \n {adata_Protein[train_idx, :].obs.cell_type.value_counts()} \n \n Test set: \n {adata_Protein[test_idx, :].obs.cell_type.value_counts()}"
)

# # Create PCA for benchmarking
# adata_merged_tmp = ad.concat([adata_RNA, adata_Protein], axis=1)
# # # adata_merged.var_names_make_unique()
# # adata_merged.obs = adata_RNA.obs
# # adata_merged.obsm = adata_RNA.obsm
# sc.tl.pca(adata_Protein)
# adata_Protein.obsm["Unintegrated"] = adata_Protein.obsm["X_pca"]


# # print("Read simulated data")
# # print(adata_Protein)
# # print("Read obsm")
# # print(list(adata_Protein.obsm.keys()))

# # bm = Benchmarker(
# #     adata_merged,
# #     batch_key="batch",
# #     label_key="cell_type",
# #     embedding_obsm_keys=["Unintegrated"],
# #     n_jobs=10,
# # )
# # bm.benchmark()
# # bm.plot_results_table(save_dir=f'./Multimodal_pretraining/plots/test.png')

# # df = bm.get_results(min_max_scale=False)
# # print(df)
