import scanpy as sc

# adata_gex = sc.read_h5ad("./Multimodal_pretraining/data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")
adata_adt = sc.read_h5ad("./Multimodal_pretraining/data/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad")
adata_adt_cite = adata_adt[:, 0:13431]
adata_adt_rna = adata_adt[:, 13431:]

# print(adata_adt.shape)
# print(adata_adt)

# print(adata_adt.obs["cell_type"])

# print("-----"*10)

# print(adata_adt.obs["cell_type"].unique())

# print("-----"*10)

# print(adata_adt.X)



print("-----"*10)

print(adata_adt_cite)
print(adata_adt_cite.X)

print("-----"*10)

print(adata_adt_rna)
print(adata_adt_rna.X)

# path = './Multimodal_pretraining/data/multi_gene_l2.loom'
# adata_RNA = sc.read(path)
# print(adata_RNA.shape)
# print(adata_RNA)

# path = './Multimodal_pretraining/data/multi_protein_l2.loom'
# adata_Protein = sc.read(path)
