import scanpy as sc
import anndata as ad

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

path = './Multimodal_pretraining/data/multi_protein_l2.loom'
adata_Protein = sc.read(path) #cell_type batch

print("Read simulated data")

# Create PCA for benchmarking
adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
# adata_merged.var_names_make_unique()
adata_merged.obs = adata_RNA.obs
# adata_merged.obsm = adata_RNA.obsm
# sc.tl.pca(adata_merged)
# adata_merged.obsm["Unintegrated"] = adata_merged.obsm["X_pca"]

print(adata_merged)
